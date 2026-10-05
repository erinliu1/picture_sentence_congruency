# Extract per-layer states at the critical word, or at the first-response position with rating instructions.
# Saves a .pt file per stimulus, naming convention as {item_index}_{image_word}_{condition}.pt
# item_index is the order of the stimulus in the sentences.json

from __future__ import annotations

import json
import random

import numpy as np

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=False)

import torch
import sys

# Reuse the behavioral prompt without importing its GPU extraction module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "behavioral"))
from gemma3_prompt import build_rating_inputs

from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

STIMULI_PATH = "/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json"
PICTURES_DIR = Path("/Intern/Erin/picture_sentence_congruency/stimuli/pictures")

with open(STIMULI_PATH, "r") as f:
    sentences = json.load(f)

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

# All Gemma 3 checkpoints are dense, run on a single GPU, and tokenize rating digits "1"-"5" as single tokens.
# Gated on Hugging Face -- HF_TOKEN's account must be approved for google/gemma-3-{4b,12b,27b}-it.
GEMMA3_DENSE_LOOKUP = {
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
    "gemma3_27b": "google/gemma-3-27b-it",
}

# Base (pretrained, non-instruction-tuned) counterparts -- "-pt" checkpoints. These ship with no
# chat template (there's no user/model turn structure to apply), so prompts are built as plain
# "<start_of_image> {sentence}" text handed straight to the processor instead of through
# apply_chat_template -- this is the usage pattern documented on the -pt model cards.
GEMMA3_BASE_LOOKUP = {
    "gemma3_4b_base": "google/gemma-3-4b-pt",
    "gemma3_12b_base": "google/gemma-3-12b-pt",
    "gemma3_27b_base": "google/gemma-3-27b-pt",
}

def extract_hidden_states(model_id, instructions=False):
    """Extract critical-word states, or first-response states with rating instructions."""
    if model_id in GEMMA3_DENSE_LOOKUP:
        model_name = GEMMA3_DENSE_LOOKUP[model_id]
        is_base = False
        dtype = torch.bfloat16
        device_map = {"": 0}
    elif model_id in GEMMA3_BASE_LOOKUP:
        model_name = GEMMA3_BASE_LOOKUP[model_id]
        is_base = True
        dtype = torch.bfloat16
        device_map = {"": 0}
    else:
        return
    if instructions and is_base:
        raise ValueError("Rating instructions require an instruction-tuned Gemma3 model.")
    title = models_lookup[model_id]["title"]

    states_folder = "hidden_states_instruction" if instructions else "hidden_states"
    HIDDEN_STATES_DIR = MODELS_DIR / model_id / states_folder
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    if not instructions:
        period_ids = processor.tokenizer.encode(".", add_special_tokens=False)
        if len(period_ids) != 1:
            raise ValueError(f"expected a single token ID for the period, but {model_id} gave {period_ids}")
        PERIOD_TOKEN_ID = period_ids[0]

    for item_index, item in enumerate(tqdm(sentences, desc=f"Extracting hidden states for {title}")):
        sentence_frame = item['sentence_frame']
        word_a, word_b = item['word_options']

        congruent_pairs = [(word_a, word_a), (word_b, word_b)]
        incongruent_pairs = [(word_a, word_b), (word_b, word_a)]
        for image_word, final_word in congruent_pairs + incongruent_pairs:
            is_congruent = (image_word == final_word)
            condition = 'congruent' if is_congruent else 'incongruent'

            sentence = f"{sentence_frame} {final_word}."
            image_path = PICTURES_DIR / f"{image_word}.png"
            image = Image.open(image_path).convert("RGB")

            if instructions:
                inputs = build_rating_inputs(processor, image, sentence)
            elif is_base:
                prompt = f"<start_of_image> {sentence}"
                inputs = processor(text=prompt, images=image, return_tensors="pt")
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": sentence,
                            },
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )

            if instructions:
                # No response digit is appended: this position predicts the first answer token.
                extraction_index = -1
            else:
                token_ids = inputs["input_ids"][0]
                period_index = (token_ids == PERIOD_TOKEN_ID).nonzero(as_tuple=True)[0][-1].item()
                extraction_index = period_index - 1

            inputs = inputs.to(model.device)
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states[1:] # exclude embedding state 0

            extracted_hidden_states = torch.stack([layer_hidden_state[0, extraction_index] for layer_hidden_state in hidden_states], dim=0).to(device="cpu", dtype=torch.float32)  # num_layers x hidden_size

            name = f"{item_index}_{image_word}_{condition}.pt"
            path = HIDDEN_STATES_DIR / name
            torch.save(extracted_hidden_states, path)

    print(f"Hidden states for {model_id} saved.")
    del model
    torch.cuda.empty_cache()

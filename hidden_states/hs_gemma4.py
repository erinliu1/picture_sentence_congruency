# For every stimulus, feed to each Gemma 4 model, pull out the model's internal hidden states at the last word of the sentence, for every layer.
# Saves a .pt file per stimulus, naming convention as {item_index}_{image_word}_{condition}.pt
# item_index is the order of the stimulus in the sentences.json

from __future__ import annotations

import json
import random

import numpy as np
import torch

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=False)

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

# All Gemma 4 checkpoints run on a single GPU and tokenize rating digits "1"-"5" as single tokens.
# Unlike Qwen, dense vs. MoE isn't a separate model class here -- Gemma4ForConditionalGeneration (or, for
# the 12B checkpoint, Gemma4UnifiedForConditionalGeneration) reads `enable_moe_block` off the checkpoint's
# own config, so AutoModelForImageTextToText dispatches to the right class either way.
# E2B/E4B additionally use per-layer embeddings (PLE) -- also just a config detail, no special-casing needed.
# Gated on Hugging Face -- HF_TOKEN's account must be approved for google/gemma-4-{E2B,E4B,12B,26B-A4B,31B}-it.
GEMMA4_DENSE_LOOKUP = {
    "gemma4_e2b": "google/gemma-4-E2B-it",
    "gemma4_e4b": "google/gemma-4-E4B-it",
    "gemma4_12b": "google/gemma-4-12B-it",
    "gemma4_31b": "google/gemma-4-31B-it",
}

GEMMA4_MOE_LOOKUP = {
    "gemma4_26b_a4b": "google/gemma-4-26B-A4B-it",
}

def extract_hidden_states(model_id):
    if model_id in GEMMA4_DENSE_LOOKUP:
        model_name = GEMMA4_DENSE_LOOKUP[model_id]
        dtype = torch.bfloat16
        device_map = {"": 0}
    elif model_id in GEMMA4_MOE_LOOKUP:
        model_name = GEMMA4_MOE_LOOKUP[model_id]
        dtype = torch.bfloat16
        device_map = {"": 0}
    else:
        return
    title = models_lookup[model_id]["title"]

    HIDDEN_STATES_DIR = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/hidden_states")
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

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

            token_ids = inputs["input_ids"][0]
            period_index = (token_ids == PERIOD_TOKEN_ID).nonzero(as_tuple=True)[0][-1].item()
            critical_word_final_token_index = period_index - 1

            inputs = inputs.to(model.device)
            with torch.inference_mode():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            hidden_states = outputs.hidden_states[1:] # exclude embedding state 0

            critical_word_hidden_states = torch.stack([layer_hidden_state[0, critical_word_final_token_index] for layer_hidden_state in hidden_states], dim=0).to(device="cpu", dtype=torch.float32)  # num_layers x hidden_size

            name = f"{item_index}_{image_word}_{condition}.pt"
            path = HIDDEN_STATES_DIR / name
            torch.save(critical_word_hidden_states, path)

    print(f"Hidden states for {model_id} saved.")
    del model
    torch.cuda.empty_cache()

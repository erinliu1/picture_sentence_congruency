# For every stimulus, feed to each PLM model, pull out the model's internal hidden states at the last word of the sentence, for every layer.
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
from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

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

# All PLM checkpoints are dense, run on a single GPU, and tokenize rating digits "1"-"5" as single tokens.
# facebook/Perception-LM-{1,3,8}B are gated on the Hub, so "local_path" (Kenny's ported copies, plain
# HF-repo-layout directories) takes priority; falls back to the Hub repo id if that directory isn't there.
PLM_DENSE_LOOKUP = {
    "plm_1b": {"local_path": "/Intern/Erin/models/Perception-LM-1B", "hf_repo_id": "facebook/Perception-LM-1B"},
    "plm_3b": {"local_path": "/Intern/Erin/models/Perception-LM-3B", "hf_repo_id": "facebook/Perception-LM-3B"},
    "plm_8b": {"local_path": "/Intern/Erin/models/Perception-LM-8B", "hf_repo_id": "facebook/Perception-LM-8B"},
}

def resolve_model_name(model_id):
    local_path = Path(PLM_DENSE_LOOKUP[model_id]["local_path"])
    return str(local_path) if local_path.is_dir() else PLM_DENSE_LOOKUP[model_id]["hf_repo_id"]

def extract_hidden_states(model_id):
    if model_id in PLM_DENSE_LOOKUP:
        model_class = PerceptionLMForConditionalGeneration
        model_name = resolve_model_name(model_id)
        dtype = torch.bfloat16
        device_map = {"": 0}
        # Only the timm-wrapped vision tower needs eager (no sdpa support yet); sdpa for the language
        # model avoids materializing the huge (num_image_tokens x num_image_tokens) fp32 attention matrix
        # that "eager" everywhere would force there, which OOMs on longer image-tile sequences.
        attn_implementation = {"text_config": "sdpa", "vision_config": "eager"}
    else:
        return
    title = models_lookup[model_id]["title"]

    HIDDEN_STATES_DIR = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/hidden_states")
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    model = model_class.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
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

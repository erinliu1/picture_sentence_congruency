# Extract critical-word or instruction-conditioned first-response states.
# Saves a .pt file per stimulus, naming convention as {item_index}_{image_word}_{condition}.pt
# item_index is the order of the stimulus in the sentences.json

from __future__ import annotations

import json
import os
import random

import numpy as np

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
# Set these before importing this module for the 235B model:
# QWEN_HS_MULTI_GPU=1 CUDA_VISIBLE_DEVICES=<three idle physical GPU indices>
MULTI_GPU = os.environ.get("QWEN_HS_MULTI_GPU") == "1"
if MULTI_GPU and len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 3:
    raise ValueError("Multi-GPU Qwen extraction requires exactly three explicitly selected GPUs.")
prepare_cuda(allow_multi_gpu=MULTI_GPU)

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "behavioral"))
from qwen_prompt import build_rating_inputs

from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration

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

QWEN_DENSE_LOOKUP = {
    "qwen3_vl_2b_instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3_vl_4b_instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3_vl_8b_instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_32b_instruct": "Qwen/Qwen3-VL-32B-Instruct",
}

QWEN_MOE_LOOKUP = {
    "qwen3_vl_30b_a3b_instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "qwen3_vl_235b_a22b_instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct", # this is the only known that will need multiple GPUs
}

MOE_SINGLE_GPU_MODELS = {"qwen3_vl_30b_a3b_instruct"} 

def extract_hidden_states(model_id, instructions=False):
    """Extract critical-word or instruction-conditioned first-response states."""
    if model_id in QWEN_DENSE_LOOKUP:
        model_class = Qwen3VLForConditionalGeneration
        model_name = QWEN_DENSE_LOOKUP[model_id]
        dtype = torch.bfloat16
        device_map = {"": 0}
    elif model_id in QWEN_MOE_LOOKUP:
        model_class = Qwen3VLMoeForConditionalGeneration
        model_name = QWEN_MOE_LOOKUP[model_id]
        if model_id in MOE_SINGLE_GPU_MODELS:
            dtype = torch.bfloat16
            device_map = {"": 0}
        else:
            if not MULTI_GPU:
                raise ValueError("Set QWEN_HS_MULTI_GPU=1 and select three idle GPUs for this model.")
            dtype = torch.bfloat16
            device_map = "auto"
    else:
        return
    title = models_lookup[model_id]["title"]

    states_folder = "hidden_states_instruction" if instructions else "hidden_states"
    HIDDEN_STATES_DIR = MODELS_DIR / model_id / states_folder
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)
    
    load_options = {}
    if device_map == "auto":
        if torch.cuda.device_count() != 3:
            raise ValueError("Expected exactly three visible CUDA devices.")
        # Leave room for vision inputs, activations, and layerwise hidden-state outputs.
        load_options["max_memory"] = {index: "175GiB" for index in range(3)}
    model = model_class.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
        **load_options,
    )
    model.eval()
    if device_map == "auto":
        placements = set(model.hf_device_map.values())
        if placements != {0, 1, 2}:
            raise RuntimeError(f"Expected placement on all three GPUs only, got {placements}")
        print(f"Model sharded across CUDA devices: {sorted(placements)}", flush=True)
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
                # The final prompt position predicts the first rating digit.
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

            # Sharded models return layers on different GPUs: move each vector before stacking.
            extracted_hidden_states = torch.stack([
                layer_hidden_state[0, extraction_index].to(device="cpu", dtype=torch.float32)
                for layer_hidden_state in hidden_states
            ], dim=0)  # num_layers x hidden_size
            if not torch.isfinite(extracted_hidden_states).all():
                raise ValueError(f"Nonfinite hidden states for {model_id}, item {item_index}.")

            name = f"{item_index}_{image_word}_{condition}.pt"
            path = HIDDEN_STATES_DIR / name
            torch.save(extracted_hidden_states, path)
    
    print(f"Hidden states for {model_id} saved.")
    del model
    torch.cuda.empty_cache()

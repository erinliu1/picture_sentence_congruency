# For every stimulus, feed to each GLM-4.6V model, pull out the model's internal hidden states at the last word of the sentence, for every layer.
# Saves a .pt file per stimulus, naming convention as {item_index}_{image_word}_{condition}.pt
# item_index is the order of the stimulus in the sentences.json
from __future__ import annotations



import json
import os
import random
import subprocess

import numpy as np

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


from prepare_cuda import prepare_cuda
MULTI_GPU = os.environ.get("GLM_HS_MULTI_GPU") == "1"
if MULTI_GPU and len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 2:
    raise ValueError("GLM_HS_MULTI_GPU=1 requires exactly two visible GPUs.")
prepare_cuda(allow_multi_gpu=MULTI_GPU)

import torch
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "behavioral"))
from glm_prompt import build_rating_inputs

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

GLM_DENSE_LOOKUP = {
    "glm_9b": "zai-org/GLM-4.6V-Flash",
}
GLM_MOE_LOOKUP = {
    "glm_106b_a12b": "zai-org/GLM-4.6V", # run this
}

def extract_hidden_states(model_id, instructions=False):
    """Extract critical-word states, or box-token states predicting behavioral ratings."""
    if model_id in GLM_DENSE_LOOKUP:
        model_name = GLM_DENSE_LOOKUP[model_id]
        dtype = torch.bfloat16
        device_map = {"": 0}
    elif model_id in GLM_MOE_LOOKUP:
        model_name = GLM_MOE_LOOKUP[model_id]
        if not MULTI_GPU or torch.cuda.device_count() != 2:
            raise ValueError("Large GLM requires GLM_HS_MULTI_GPU=1 and two visible GPUs.")
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        return
    title = models_lookup[model_id]["title"]

    states_folder = "hidden_states_instruction" if instructions else "hidden_states"
    HIDDEN_STATES_DIR = MODELS_DIR / model_id / states_folder
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
        **({"max_memory": {0: "175GiB", 1: "175GiB"}} if model_id in GLM_MOE_LOOKUP else {})
    )
    if model_id in GLM_MOE_LOOKUP:
        placements = set(model.hf_device_map.values())
        if placements != {0, 1}:
            raise ValueError(f"Expected placement on both GPUs only, got {placements}.")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    if instructions:
        box_token_id = processor.tokenizer.convert_tokens_to_ids("<|begin_of_box|>")
        if box_token_id is None or box_token_id == processor.tokenizer.unk_token_id:
            raise ValueError("Missing GLM box token.")
        rating_token_ids = [processor.tokenizer.encode(str(i), add_special_tokens=False) for i in range(1, 6)]
        if any(len(ids) != 1 for ids in rating_token_ids):
            raise ValueError("Expected single-token rating digits.")
        rating_token_ids = [ids[0] for ids in rating_token_ids]
    else:
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
                inputs = build_rating_inputs(processor, image, sentence).to(model.device)
                generation_options = {}
                if model_id in GLM_MOE_LOOKUP:
                    prompt_len = inputs["input_ids"].shape[-1]
                    def prefix_allowed_tokens_fn(batch_id, input_ids):
                        return [box_token_id] if input_ids.shape[-1] == prompt_len else rating_token_ids
                    generation_options = {"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn, "output_logits": True}
                with torch.inference_mode():
                    gen = model.generate(
                        **inputs,
                        max_new_tokens=2,
                        do_sample=False,
                        output_scores=True,
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        **generation_options,
                    )
                if model_id in GLM_MOE_LOOKUP and not torch.isfinite(gen.logits[1]).all():
                    raise ValueError(f"Nonfinite rating logits for {model_id}, item {item_index}.")
                generated_tokens = gen.sequences[0, inputs["input_ids"].shape[1]:]
                if len(generated_tokens) < 2 or int(generated_tokens[0]) != box_token_id:
                    raise ValueError(f"Expected box token then rating for {model_id}, item {item_index}.")
                # Generation step 1 consumes the first generated token (the box marker).
                # These states predict the digit scored by behavioral gen.scores[1].
                layer_states = gen.hidden_states[1][1:]
                extraction_index = -1
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
                layer_states = outputs.hidden_states[1:]


            extracted_hidden_states = torch.stack([layer_hidden_state[0, extraction_index].to(device="cpu", dtype=torch.float32) for layer_hidden_state in layer_states], dim=0)  # num_layers x hidden_size

            if not torch.isfinite(extracted_hidden_states).all():
                raise ValueError(f"Nonfinite hidden states for item {item_index}.")

            name = f"{item_index}_{image_word}_{condition}.pt"
            path = HIDDEN_STATES_DIR / name
            torch.save(extracted_hidden_states, path)

    print(f"Hidden states for {model_id} saved.")
    del model
    torch.cuda.empty_cache()

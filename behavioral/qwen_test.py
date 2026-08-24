
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # Set the GPU device to use

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=True)

import json
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLMoeForConditionalGeneration

import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

STIMULI_PATH = Path("/Intern/Erin/picture_sentence_congruency/stimuli")
PICTURES_DIR = Path("/Intern/Erin/picture_sentence_congruency/stimuli/pictures")

with open(STIMULI_PATH / "sentences.json", "r") as f:
    sentences = json.load(f)

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

models = {
    "qwen3_vl_235b_a22b_instruct": {
        "class": Qwen3VLMoeForConditionalGeneration,
        "name": "Qwen/Qwen3-VL-235B-A22B-Instruct",
        "title": "Qwen3-VL-235B-A22B-Instruct",
        "family": "qwen"
    },
}

SYSTEM_PROMPT = """
You will be shown one picture and one sentence. Your task is to judge how compatible the final word of the sentence is with the situation shown in the picture.

When making your judgment:
* Focus on the final word of the sentence. Treat the rest of the sentence as given.
* The picture does not need to prove that the sentence ending is the only possible interpretation. Simply, decide whether the final word is a natural and reasonable interpretation of the picture.

Use the following rating scale:
1 - Not compatible at all. The final word is clearly inconsistent with / unrelated to / contradicts the picture.
2 - Low compatibility. The final word is technically plausible but would not be commonly associated with the picture / it would be unusual or difficult to reconcile this sentence with the picture.
3 - Uncertain. The final word provides weak / ambiguous / mixed evidence for the picture.
4 - Compatible. The final word makes the sentence a reasonable interpretation of the picture.
5 - Very compatible. The final word makes the sentence very consistent with / a natural interpretation of the picture.

Respond with exactly one token: 1, 2, 3, 4, or 5. Do not output any additional text.
""".strip()

model_id = "qwen3_vl_235b_a22b_instruct"
title = models[model_id]["title"]

model = models[model_id]["class"].from_pretrained(
    models[model_id]["name"],
    dtype="auto",
    device_map="auto",
    attn_implementation="sdpa"
)
model.eval()
processor = AutoProcessor.from_pretrained(models[model_id]["name"])

RATING_TOKEN_IDS = []
for rating in range(1, 6):
    rating_ids = processor.tokenizer.encode(str(rating), add_special_tokens=False)
    if len(rating_ids) != 1:
        raise ValueError(f"expected a single token ID for the digit {rating!r}, but model gave {rating_ids}")
    RATING_TOKEN_IDS.append(rating_ids[0])

expected_ratings = []
test_stimuli = sentences[:2]

for item_index, item in enumerate(test_stimuli):
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

        print(f"Item {item_index}. Sentence: {sentence} Image: {image_word}. Condition: {condition}")
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                    }
                ],
            },
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
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # Only take the newly generated tokens, excluding the input prompt
        new_tokens = generated_ids[:, prompt_len:]

        response = processor.tokenizer.decode(
            new_tokens[0],
            skip_special_tokens=True,
        )

        print(f"Model response: {response!r}")
        print("-" * 80)

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=False)

import json
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

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

# GLM-4.6V checkpoints tokenize rating digits "1"-"5" as single tokens, but sometimes output a box-marker token "<|begin_of_box|>" before the rating digit.
GLM_DENSE_LOOKUP = {
    "glm_9b": "zai-org/GLM-4.6V-Flash",
}
GLM_MOE_LOOKUP = {
    "glm_106b_a12b": "zai-org/GLM-4.6V", # broken because it needs multiple GPUs
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

def extract_ratings(model_id):
    if model_id in GLM_DENSE_LOOKUP:
        model_name = GLM_DENSE_LOOKUP[model_id]
        dtype = torch.bfloat16
        device_map = {"": 0}
    # elif model_id in GLM_MOE_LOOKUP:
    #     model_name = GLM_MOE_LOOKUP[model_id]
    #     dtype = "auto"
    #     device_map = "auto" broken
    else:
        return
    title = models_lookup[model_id]["title"]

    BEHAVIOR_DIR = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/behavior")
    if BEHAVIOR_DIR.exists() and (BEHAVIOR_DIR / "ratings.csv").exists():
        print(f"Ratings for {model_id} already exist, skipping")
        return

    BEHAVIOR_DIR.mkdir(parents=True, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    RATING_TOKEN_IDS = []
    for rating in range(1, 6):
        rating_ids = processor.tokenizer.encode(str(rating), add_special_tokens=False)
        if len(rating_ids) != 1:
            raise ValueError(f"expected a single token ID for the digit {rating!r}, but model gave {rating_ids}")
        RATING_TOKEN_IDS.append(rating_ids[0])

    box_token_id = processor.tokenizer.convert_tokens_to_ids("<|begin_of_box|>")

    expected_ratings = []
    for item_index, item in enumerate(tqdm(sentences, desc=f"Extracting ratings for {title}")):
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
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=2,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            generated_tokens = gen.sequences[0, inputs["input_ids"].shape[1]:]
            used_box_position = int(generated_tokens[0].item()) == box_token_id
            next_token_logits = gen.scores[1][0] if used_box_position else gen.scores[0][0]

            rating_logits = next_token_logits[RATING_TOKEN_IDS]
            probabilities = F.softmax(rating_logits, dim=0)
            ratings = torch.arange(1, 6, device=probabilities.device, dtype=probabilities.dtype)
            expected_rating = (probabilities * ratings).sum().item()
            max_probability = probabilities.max().item()
            top_one_token = int(ratings[probabilities.argmax()].item())
            full_probabilities = F.softmax(next_token_logits, dim=0)
            total_probability_mass = full_probabilities[RATING_TOKEN_IDS].sum().item()
            expected_ratings.append({
                "item_index": item_index,
                "image_word": image_word,
                "condition": condition,
                "rating": expected_rating,
                "max_probability": max_probability,
                "top_one_token": top_one_token,
                "p1": probabilities[0].item(),
                "p2": probabilities[1].item(),
                "p3": probabilities[2].item(),
                "p4": probabilities[3].item(),
                "p5": probabilities[4].item(),
                "total_probability_mass": total_probability_mass,
                "used_box_position": used_box_position,
            })

    df = pd.DataFrame(expected_ratings)
    df = df.sort_values(by=['item_index', 'condition', 'image_word']).reset_index(drop=True)
    df.to_csv(BEHAVIOR_DIR / "ratings.csv", index=False)

    print(f"Ratings for {model_id} saved.")
    del model
    torch.cuda.empty_cache()

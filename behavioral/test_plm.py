# Small smoke test for the PLM (Perception Language Model) series before writing a full
# behavioral_plm.py. Loads facebook/Perception-LM-1B, verifies rating digits "1"-"5" tokenize
# as single tokens, runs a couple of picture-sentence pairs through the model, and prints
# accuracy/health diagnostics inline (no plots, no full 320-row ratings.csv).
#
# Modeled on behavioral_intern.py / behavioral_qwen.py for the rating-extraction logic, and on
# kenny_stuff/Neuroscience/src/neuro_congruency/models/plm.py for how to load/prompt PLM:
# - model class is PerceptionLMForConditionalGeneration (native transformers support)
# - attn_implementation="eager" -- PLM's timm-wrapped vision tower doesn't support sdpa yet
# - PLM's chat_template.jinja special-cases the system message as `messages[0]['content']|trim`,
#   a plain string -- unlike every other role, which expects the structured content-list format.
#   Passing the list format (as behavioral_intern.py/behavioral_qwen.py do) renders as the raw
#   Python repr of the list instead of the instruction text, so the system message must be a bare
#   string here.

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=False)

import json
import torch
from PIL import Image
from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

import torch.nn.functional as F
import pandas as pd

STIMULI_PATH = Path("/Intern/Erin/picture_sentence_congruency/stimuli")
PICTURES_DIR = Path("/Intern/Erin/picture_sentence_congruency/stimuli/pictures")

with open(STIMULI_PATH / "sentences.json", "r") as f:
    sentences = json.load(f)

# Erin ported Kenny's PLM checkpoints to plain local directories (flat HF-repo layout: config.json +
# model.safetensors* + tokenizer files) -- loadable directly via from_pretrained(local_dir), same as
# resolve_model_id() in kenny_stuff/Neuroscience/src/neuro_congruency/models/base.py does. This also
# sidesteps facebook/Perception-LM-1B being a gated Hub repo (access request was still pending review).
PLM_LOCAL_PATHS = {
    "1B": "/Intern/Erin/Perception-LM-1B",
    "3B": "/Intern/Erin/Perception-LM-3B",
    "8B": "/Intern/Erin/Perception-LM-8B",
}
MODEL_NAME = PLM_LOCAL_PATHS["1B"]
N_TEST_ITEMS = 3  # how many of the 80 sentence-frame stimuli to test (x4 congruent/incongruent pairs each)

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


def main():
    print(f"Loading {MODEL_NAME} ...")
    model = PerceptionLMForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="eager",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # --- verify rating digits 1-5 tokenize as single tokens ---
    RATING_TOKEN_IDS = []
    print("\nVerifying rating digit tokenization:")
    for rating in range(1, 6):
        rating_ids = processor.tokenizer.encode(str(rating), add_special_tokens=False)
        status = "OK" if len(rating_ids) == 1 else "FAIL"
        print(f"  '{rating}' -> token ids {rating_ids} [{status}]")
        if len(rating_ids) != 1:
            raise ValueError(f"expected a single token ID for the digit {rating!r}, but model gave {rating_ids}")
        RATING_TOKEN_IDS.append(rating_ids[0])

    # --- run a couple of picture-sentence pairs ---
    expected_ratings = []
    test_items = list(enumerate(sentences))[:N_TEST_ITEMS]
    print(f"\nRunning {N_TEST_ITEMS} stimulus item(s) x 4 congruent/incongruent pairs = {N_TEST_ITEMS * 4} rows...")
    for item_index, item in test_items:
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
                # PLM's chat template renders messages[0]['content'] as a bare string for the
                # system role (unlike the structured content-list every other role expects) --
                # see module docstring above.
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
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
            )

            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs, use_cache=False)

            next_token_logits = outputs.logits[0, -1]
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
                "final_word": final_word,
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
            })

    df = pd.DataFrame(expected_ratings)
    df = df.sort_values(by=['item_index', 'condition', 'image_word']).reset_index(drop=True)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n--- Raw ratings ---")
    print(df[["item_index", "image_word", "final_word", "condition", "rating", "top_one_token", "max_probability", "total_probability_mass"]])

    # --- accuracy diagnostic (same rule as behavioral_accuracy.compute_accuracy) ---
    y_true = df["condition"] == "congruent"
    y_pred = df["rating"] > 3
    accuracy = (y_true == y_pred).mean()
    print(f"\n--- Accuracy (rating > 3 vs. congruent, n={len(df)}) ---")
    print(f"accuracy = {accuracy:.3f}")

    # --- health diagnostic (same summary as behavioral_health.extract_health) ---
    print("\n--- Health (per condition) ---")
    for condition in ["congruent", "incongruent"]:
        condition_df = df.query("condition == @condition")
        print(f"[{condition}] mean_rating={condition_df['rating'].mean():.3f} "
              f"std_rating={condition_df['rating'].std():.3f} "
              f"mean_top_one_token={condition_df['top_one_token'].mean():.3f}")
        proportions = {v: (condition_df["top_one_token"] == v).mean() for v in range(1, 6)}
        print(f"           top_one_token proportions: {proportions}")

    overall_top_one_is_1 = (df["top_one_token"] == 1).mean()
    print(f"\noverall proportion of rows with top_one_token == 1: {overall_top_one_is_1:.3f}")
    if overall_top_one_is_1 > 0.8:
        print("-> Matches Kenny's prior finding: PLM-1B collapses to top_one_token=1 on almost every item (poor health).")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

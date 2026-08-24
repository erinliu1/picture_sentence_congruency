import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, pearsonr, spearmanr

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
STIMULI_PATH = Path("/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json")
BEHAVIORAL_RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")

with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

with open(STIMULI_PATH, "r") as f:
    sentences = json.load(f)

def compute_grounding_correlation(model_id):
    """For each final word, pair its rating when the picture matches (congruent) against its
    rating when the picture doesn't match (incongruent, same final word). Pearson r across all
    such pairs: near 1 = the picture doesn't change the rating (model ignores the image);
    near 0 = the picture strongly changes the rating (model uses the image)."""
    df = pd.read_csv(MODELS_DIR / model_id / "behavior" / "ratings.csv")
    pairs = []
    for item_index, item in enumerate(sentences):
        word_a, word_b = item["word_options"]
        sub = df[df["item_index"] == item_index]
        for w_match, w_mismatch in [(word_a, word_b), (word_b, word_a)]:
            matched = sub[(sub["image_word"] == w_match) & (sub["condition"] == "congruent")]["rating"]
            mismatched = sub[(sub["image_word"] == w_mismatch) & (sub["condition"] == "incongruent")]["rating"]
            if len(matched) and len(mismatched):
                pairs.append((matched.values[0], mismatched.values[0]))
    pdf = pd.DataFrame(pairs, columns=["matched", "mismatched"])
    return pdf["matched"].corr(pdf["mismatched"])


def extract_grounding():
    acc_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "accuracy.csv").set_index("source")
    corr_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "correlation.csv").set_index("source")
    rows = []
    for model_id, info in models_lookup.items():
        if model_id not in corr_df.index:
            continue
        rows.append({
            "source": model_id,
            "grounding_r": compute_grounding_correlation(model_id),
        })
    df = pd.DataFrame(rows)
    df.to_csv(BEHAVIORAL_RESULTS_DIR / "grounding.csv", index=False)
    return df

df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "grounding.csv").set_index("source")
# sort by highest grounding_r
df = df.sort_values("grounding_r", ascending=False)
print(df.head(10))
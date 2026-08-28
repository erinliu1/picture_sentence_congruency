"""
For a given model:
  1. Take the FINAL layer (max layer_index in models/{model_id}/alignment/coefficients.csv),
     instead of the highest-coefficient layer (see make_max_layer_pairs.py for that version).
  2. Pull probability_congruent for every item at that layer from models/{model_id}/decoding/results.csv.
  3. Pull rating for every item from models/{model_id}/behavior/ratings.csv.
  4. Save the merged table to models/{model_id}/alignment/final_layer_pairs.csv with columns:
     stimulus_index, image_word, condition, rating, probability_congruent

Hand-edit model_id below before running.
"""

import pandas as pd
from pathlib import Path

model_id = "internvl3_78b_instruct"

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
MODEL_DIR = MODELS_DIR / model_id

coefficients_path = MODEL_DIR / "alignment" / "coefficients.csv"
results_path = MODEL_DIR / "decoding" / "results.csv"
ratings_path = MODEL_DIR / "behavior" / "ratings.csv"
output_path = MODEL_DIR / "alignment" / "final_layer_pairs.csv"

# 1. Take the final layer.
coefficients = pd.read_csv(coefficients_path)
final_row = coefficients.loc[coefficients["layer_index"].idxmax()]
final_layer = int(final_row["layer_index"])
print(f"Final layer: {final_layer} (coefficient={final_row['coefficient']})")

# 2. Extract probability_congruent for every item at that layer.
results = pd.read_csv(results_path)
layer_results = results.loc[
    results["layer_index"] == final_layer,
    ["item_index", "image_word", "condition", "probability_congruent"],
]

# 3. Extract rating for every item.
ratings = pd.read_csv(ratings_path)[["item_index", "image_word", "condition", "rating"]]

# 4. Merge on item_index/image_word/condition, rename item_index -> stimulus_index.
merged = layer_results.merge(
    ratings, on=["item_index", "image_word", "condition"], how="left"
)
merged = merged.rename(columns={"item_index": "stimulus_index"})
merged = merged[["stimulus_index", "image_word", "condition", "rating", "probability_congruent"]]
merged = merged.sort_values("stimulus_index").reset_index(drop=True)

merged.to_csv(output_path, index=False)
print(f"Saved {len(merged)} rows to {output_path}")

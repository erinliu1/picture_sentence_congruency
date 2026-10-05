"""Build final-layer decoder probability/behavioral rating pairs without import-time work."""

from pathlib import Path

import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")


def make_final_layer_pairs(model_id, instructions=False):
    """Save and return final-layer pairs for the selected decoding/alignment mode."""
    model_dir = MODELS_DIR / model_id
    alignment_folder = "alignment_instruction" if instructions else "alignment"
    decoding_folder = "decoding_instruction" if instructions else "decoding"
    coefficients_path = model_dir / alignment_folder / "coefficients.csv"
    results_path = model_dir / decoding_folder / "results.csv"
    ratings_path = model_dir / "behavior" / "ratings.csv"
    output_path = model_dir / alignment_folder / "final_layer_pairs.csv"

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
    return merged

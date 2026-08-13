import json
from pathlib import Path

import numpy as np
import pandas as pd

# Item-resampling bootstrap CIs for layer-wise decoding accuracy (B=10,000) at the stimulus-set level.

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

N_BOOTSTRAPS = 10_000
N_ITEMS = 80
SEED = 0

def per_item_correct_counts(results_df):
    # For every layer and every stimulus set (item_index), count how many of its 2 congruent examples and 2 incongruent examples were classified correctly.

    layers = np.sort(results_df["layer_index"].unique())

    correct = results_df["pass"].to_numpy() == "✅"
    is_congruent = results_df["condition"].to_numpy() == "congruent"

    congruent_correct = np.zeros((len(layers), N_ITEMS), dtype=np.uint8)
    incongruent_correct = np.zeros((len(layers), N_ITEMS), dtype=np.uint8)

    layer_index_of_row = results_df["layer_index"].to_numpy()
    item_index_of_row = results_df["item_index"].to_numpy()
    layer_pos = np.searchsorted(layers, layer_index_of_row)

    np.add.at(congruent_correct, (layer_pos[is_congruent], item_index_of_row[is_congruent]), correct[is_congruent])
    np.add.at(incongruent_correct, (layer_pos[~is_congruent], item_index_of_row[~is_congruent]), correct[~is_congruent])

    congruent_counts = np.zeros((len(layers), N_ITEMS), dtype=np.uint8)
    incongruent_counts = np.zeros((len(layers), N_ITEMS), dtype=np.uint8)
    np.add.at(congruent_counts, (layer_pos[is_congruent], item_index_of_row[is_congruent]), 1)
    np.add.at(incongruent_counts, (layer_pos[~is_congruent], item_index_of_row[~is_congruent]), 1)

    if not np.all(congruent_counts == 2) or not np.all(incongruent_counts == 2):
        raise ValueError(
            "Expected exactly 2 congruent and 2 incongruent observations per "
            "item per layer; results.csv does not match that shape."
        )

    return layers, congruent_correct, incongruent_correct

def lr_bootstrap(model_id):
    model_dir = MODELS_DIR / model_id
    results_dir = model_dir / "decoding"
    classifier_results_df = pd.read_csv(results_dir / "results.csv")
    
    if (results_dir / 'all_bootstraps.csv').exists():
        print(f"Bootstraps already exist for {model_id}, skipping bootstrap.")
        return

    layers, congruent_correct, incongruent_correct = per_item_correct_counts(classifier_results_df)

    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, N_ITEMS, size=(N_BOOTSTRAPS, N_ITEMS))

    rows = []
    for layer_pos, layer_index in enumerate(layers):
        resampled_congruent = congruent_correct[layer_pos][idx]
        resampled_incongruent = incongruent_correct[layer_pos][idx]

        congruent_accuracy = resampled_congruent.sum(axis=1) / (N_ITEMS * 2)
        incongruent_accuracy = resampled_incongruent.sum(axis=1) / (N_ITEMS * 2)
        all_accuracy = (resampled_congruent.sum(axis=1) + resampled_incongruent.sum(axis=1)) / (N_ITEMS * 4)

        rows.append(pd.DataFrame({
            "SEED": np.arange(N_BOOTSTRAPS),
            "layer_index": layer_index,
            "all_accuracy": all_accuracy,
            "congruent_accuracy": congruent_accuracy,
            "incongruent_accuracy": incongruent_accuracy,
        }))

    all_bootstraps_df = pd.concat(rows, ignore_index=True).sort_values(by=["SEED", "layer_index"]).reset_index(drop=True)
    all_bootstraps_df.to_csv(results_dir / "all_bootstraps.csv", index=False)
    print(f"Bootstraps for {model_id} saved.")
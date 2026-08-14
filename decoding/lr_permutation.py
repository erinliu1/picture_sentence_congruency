import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

CV_DIR = "/Intern/Erin/picture_sentence_congruency/decoding/cv"

with open(f"{CV_DIR}/cv_lookup_pt.json", "r") as f:
    cv_lookup = json.load(f)

N_FOLDS = len(cv_lookup)

# Set per model, before creating each model's ProcessPoolExecutor, so the
# forked workers inherit them already populated (see the __main__ loop).
ALL_HIDDEN_STATES = {}
# fold -> {"train": {...}, "test": {...}}, each {"X", "item_index",
# "base_label"}. The hidden-state values and fold membership never change
# across permutation seeds (only which items get their label flipped does),
# so these are built once per model in run_all_permutations and reused by
# every seed/worker instead of being re-gathered from ALL_HIDDEN_STATES on
# every single permutation.
FOLD_DATA = {}
PERMUTATIONS_DIR = None

def unpack_item(item):
    parts = item.split("_")
    item_index = int(parts[0])
    image_word = parts[1]
    condition = parts[2].split(".")[0]
    return item_index, image_word, condition

def get_label(condition, reverse=False):
    if reverse:
        if condition == 'congruent':
            return 0
        else:
            return 1
    else:
        if condition == 'congruent':
            return 1
        else:
            return 0

def build_fold_data(items_dict):
    # items_dict: {item_index: [pt_filename, ...]}, e.g. cv_lookup[fold]["train"].
    # Returns the stacked hidden states plus, per row, the source item_index
    # and its non-reversed ("base") label, so a seed's reverse flip can later
    # be applied with a single vectorized where() instead of a Python loop.
    X_rows, item_indices, base_labels = [], [], []
    for item_index, pt_list in items_dict.items():
        for item_name in pt_list:
            _, _, condition = unpack_item(item_name)
            X_rows.append(ALL_HIDDEN_STATES[item_name])
            item_indices.append(int(item_index))
            base_labels.append(get_label(condition, reverse=False))
    return {
        "X": np.stack(X_rows, axis=0),
        "item_index": np.asarray(item_indices, dtype=np.int64),
        "base_label": np.asarray(base_labels, dtype=np.int64),
    }

def compute_layerwise_accuracy(predictions):
    # predictions: list of (layer_index, condition, passed) tuples
    predictions_df = pd.DataFrame(predictions, columns=["layer_index", "condition", "pass"])

    layerwise_accuracy = []
    for layer_index in sorted(predictions_df["layer_index"].unique()):
        layer_df = predictions_df[predictions_df["layer_index"] == layer_index]
        accuracy = (layer_df["pass"] == "✅").mean()
        congruent_df = layer_df[layer_df["condition"] == "congruent"]
        congruent_accuracy = (congruent_df["pass"] == "✅").mean()
        incongruent_df = layer_df[layer_df["condition"] == "incongruent"]
        incongruent_accuracy = (incongruent_df["pass"] == "✅").mean()
        layerwise_accuracy.append({
            "layer_index": layer_index,
            "all_accuracy": accuracy,
            "congruent_accuracy": congruent_accuracy,
            "incongruent_accuracy": incongruent_accuracy,
        })
    return pd.DataFrame(layerwise_accuracy)

def run_permutation(seed):
    # Checkpoint per seed. returns per-layer accuracy
    checkpoint_path = f"{PERMUTATIONS_DIR}/{seed}.csv"
    if os.path.exists(checkpoint_path):
        return pd.read_csv(checkpoint_path)

    rng = np.random.default_rng(seed)
    # reverse_by_item[item_index] tells whether that item's label is flipped
    # for this seed; indexing it with a fold's item_index array (fancy
    # indexing) is a drop-in vectorized replacement for the old
    # "int(item_index) in reverse_indices" set-membership check per row.
    reverse_by_item = rng.random(80) < 0.5

    predictions = []

    for fold in range(1, N_FOLDS + 1):
        train = FOLD_DATA[fold]["train"]
        test = FOLD_DATA[fold]["test"]

        X = train["X"] # (288, 36, 4096) -- identical across all seeds
        train_reverse = reverse_by_item[train["item_index"]]
        y = np.where(train_reverse, 1 - train["base_label"], train["base_label"])

        X_test = test["X"] # (n_test, 36, 4096) -- identical across all seeds
        test_reverse = reverse_by_item[test["item_index"]]
        y_test = np.where(test_reverse, 1 - test["base_label"], test["base_label"])
        test_conditions = np.where(y_test == 1, 'congruent', 'incongruent')

        N_LAYERS = X.shape[1]

        for layer_index in range(N_LAYERS):
            X_layer = X[:, layer_index, :] # (288, 4096)

            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.01,
                    solver="liblinear",
                    max_iter=10_000,
                    random_state=seed,
                ),
            )

            classifier.fit(X_layer, y)

            X_test_layer = X_test[:, layer_index, :] # (n_test, 4096)
            predicted_labels = classifier.predict(X_test_layer) # (n_test,)
            passed = np.where(predicted_labels == y_test, '✅', '❌')

            predictions.extend(zip(
                [layer_index] * len(passed), test_conditions, passed
            ))

    accuracy_df = compute_layerwise_accuracy(predictions)
    accuracy_df.insert(0, "seed", seed)
    accuracy_df.to_csv(checkpoint_path, index=False)
    return accuracy_df

def run_all_permutations(model_id, n_permutations):
    global PERMUTATIONS_DIR

    title = models_lookup[model_id]["title"]
    model_dir = MODELS_DIR / model_id

    hidden_states_dir = model_dir / "hidden_states"

    ALL_HIDDEN_STATES.clear()
    for filename in os.listdir(hidden_states_dir):
        ALL_HIDDEN_STATES[filename] = torch.load(
            hidden_states_dir / filename, map_location="cpu", weights_only=False
        ).float().numpy()

    FOLD_DATA.clear()
    for fold in range(1, N_FOLDS + 1):
        FOLD_DATA[fold] = {
            "train": build_fold_data(cv_lookup[str(fold)]["train"]),
            "test": build_fold_data(cv_lookup[str(fold)]["test"]),
        }

    decoding_dir = model_dir / "decoding"
    decoding_dir.mkdir(exist_ok=True)
    PERMUTATIONS_DIR = str(decoding_dir / "permutations")
    os.makedirs(PERMUTATIONS_DIR, exist_ok=True)

    # A fresh pool per model: workers are forked here, after
    # ALL_HIDDEN_STATES/FOLD_DATA/PERMUTATIONS_DIR are set for this model, so
    # they inherit them already populated.
    # submit()/as_completed() (rather than executor.map) so the progress bar
    # advances as each seed actually finishes -- including checkpointed seeds
    # from a prior interrupted run, which return almost immediately -- rather
    # than sitting at 0% until every one of the n_permutations is done.
    with ProcessPoolExecutor(max_workers=70) as executor:
        futures = [executor.submit(run_permutation, seed) for seed in range(n_permutations)]
        seed_accuracy_dfs = [
            future.result()
            for future in tqdm(
                as_completed(futures), total=n_permutations, desc=f"Permutations for {title}"
            )
        ]

    permutation_layerwise_accuracies_df = (
        pd.concat(seed_accuracy_dfs, ignore_index=True)
        .sort_values(by=["seed", "layer_index"])
        .reset_index(drop=True)
    )
    permutation_layerwise_accuracies_df.to_csv(
        decoding_dir / "permutation_layerwise_accuracies.csv", index=False
    )

    max_stats_df = (
        permutation_layerwise_accuracies_df
        .groupby("seed", as_index=False)["all_accuracy"]
        .max()
        .rename(columns={"all_accuracy": "max_accuracy"})
    )
    max_stats_df.to_csv(decoding_dir / "max_stats.csv", index=False)
    print(f"Permutation results for {model_id} saved.")
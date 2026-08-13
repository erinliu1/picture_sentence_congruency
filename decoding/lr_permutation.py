import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

def get_item(pt_list, reverse=False):
    X, y = [], []
    for item_name in pt_list:
        item_index, image_word, condition = unpack_item(item_name)
        X.append(ALL_HIDDEN_STATES[item_name])
        y.append(get_label(condition, reverse=reverse))
    return X, y

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
    item_indices = np.arange(80)
    reverse_indices = set(item_indices[rng.random(len(item_indices)) < 0.5])

    predictions = []

    for fold in range(1, N_FOLDS + 1):
        train_items = cv_lookup[str(fold)]["train"]
        test_items = cv_lookup[str(fold)]["test"]

        X_train, y_train = [], []
        for item_index, pt_list in train_items.items():
            reverse = int(item_index) in reverse_indices
            X_item, y_item = get_item(pt_list, reverse=reverse)
            X_train.extend(X_item)
            y_train.extend(y_item)

        X = np.stack(X_train, axis=0) # (288, 36, 4096)
        y = np.asarray(y_train, dtype=np.int64) # (288,)

        classifiers = {}
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
            classifiers[layer_index] = classifier

        for item_index, pt_list in test_items.items():
            reverse = int(item_index) in reverse_indices
            for item_name in pt_list:
                _, image_word, condition = unpack_item(item_name)
                X_test = ALL_HIDDEN_STATES[item_name] # (36, 4096)
                label = get_label(condition, reverse=reverse)
                condition = 'congruent' if label == 1 else 'incongruent'

                for layer_index, classifier in classifiers.items():
                    X_test_layer = X_test[layer_index, :].reshape(1, -1) # (1, 4096)
                    predicted_label = int(classifier.predict(X_test_layer)[0])
                    predictions.append((
                        layer_index,
                        condition,
                        '✅' if predicted_label == label else '❌',
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

    decoding_dir = model_dir / "decoding"
    decoding_dir.mkdir(exist_ok=True)
    PERMUTATIONS_DIR = str(decoding_dir / "permutations")
    os.makedirs(PERMUTATIONS_DIR, exist_ok=True)

    # A fresh pool per model: workers are forked here, after
    # ALL_HIDDEN_STATES/PERMUTATIONS_DIR are set for this model, so they
    # inherit them already populated.
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 2) as executor:
        seed_accuracy_dfs = list(executor.map(run_permutation, range(n_permutations)))

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
    print(f"Saved permutation results to {decoding_dir / 'permutation_layerwise_accuracies.csv'} and {decoding_dir / 'max_stats.csv'}")
"""Refit and save RR probes at every decoded model layer.

Usage: python save_steering_probes.py

Uses the same CV item order, labels, float32 features, and fitted
StandardScaler + LogisticRegression settings as lr_train.py. Outputs live in
models/<model_id>/steering/layer<layer>/fold<fold>/probe.joblib. Existing
outputs are protected.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_vl_8b_instruct"
sys.path.insert(0, str(ROOT_DIR / "decoding"))
from lr_train import get_label, unpack_item

CV_PATH = ROOT_DIR / "decoding/cv/cv_lookup_pt.json"
SEED = 42


def probe_path(model_dir, layer, fold):
    """Return the canonical path for one layer/fold steering probe."""
    return model_dir / "steering" / f"layer{layer}" / f"fold{fold}" / "probe.joblib"


def save_steering_probes(model_id, layers=None):
    model_dir = ROOT_DIR / "models" / model_id
    accuracy_path = model_dir / "decoding_instruction/layerwise_accuracy.csv"
    accuracy = pd.read_csv(accuracy_path).sort_values("layer_index").reset_index(drop=True)
    if accuracy.empty or accuracy[["layer_index", "all_accuracy"]].isna().any().any():
        raise ValueError(f"Empty or incomplete accuracy data: {accuracy_path}")
    if accuracy.layer_index.duplicated().any():
        raise ValueError("Duplicate layer indices")
    final_layer = int(accuracy.layer_index.max())
    available_layers = accuracy.layer_index.astype(int).tolist()
    if available_layers != list(range(final_layer + 1)):
        raise ValueError(f"Expected contiguous layers 0 through {final_layer}: {available_layers}")
    layers = available_layers if layers is None else [int(layer) for layer in layers]
    if not layers or len(set(layers)) != len(layers) or not set(layers) <= set(available_layers):
        raise ValueError(f"Invalid selected layers: {layers}")
    cv_bytes = CV_PATH.read_bytes()
    cv_lookup = json.loads(cv_bytes)
    if set(cv_lookup) != {str(fold) for fold in range(1, 11)}:
        raise ValueError("Expected CV folds 1 through 10")
    output_dir = model_dir / "steering"
    if any(path for layer in layers for path in (output_dir / f"layer{layer}").rglob("*.joblib")):
        raise FileExistsError(f"Probe outputs already exist: {output_dir}")

    # Load every layer once; assemble examples in each fold's original nested
    # dict/list order without altering their feature values.
    filenames = dict.fromkeys(
        name for split in cv_lookup.values() for partition in ("train", "test")
        for names in split[partition].values() for name in names
    )
    features, labels = {}, {}
    for name in tqdm(filenames, desc=f"Loading {model_id} RR states"):
        state = torch.load(model_dir / "hidden_states_instruction" / name,
                           map_location="cpu", weights_only=False)
        if state.ndim != 2 or state.shape[0] != final_layer + 1:
            raise ValueError(f"Unexpected hidden-state shape for {name}: {state.shape}")
        all_layers = state.float().numpy()
        if not np.isfinite(all_layers).all():
            raise ValueError(f"Nonfinite hidden states: {name}")
        features[name] = all_layers
        labels[name] = get_label(unpack_item(name)[2])

    original = pd.read_csv(model_dir / "decoding_instruction/results.csv")
    original["item_index"] = original.item_index.astype(int)
    original = original.set_index(["layer_index", "item_index", "image_word", "condition"])
    if not original.index.is_unique:
        raise ValueError("Duplicate rows in original decoding results")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Selected layers: {layers}", flush=True)
    for fold in tqdm(range(1, 11), desc="Fitting steering probes"):
        split = cv_lookup[str(fold)]
        train_names = [name for names in split["train"].values() for name in names]
        test_names = [name for names in split["test"].values() for name in names]
        if set(split["train"]) & set(split["test"]):
            raise ValueError(f"Train/test stimulus-set overlap in fold {fold}")
        X_train = np.stack([features[name] for name in train_names])
        y_train = np.array([labels[name] for name in train_names], dtype=np.int64)
        X_test = np.stack([features[name] for name in test_names])
        y_test = np.array([labels[name] for name in test_names], dtype=np.int64)
        for layer in layers:
            pipeline = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.01, solver="liblinear", max_iter=10_000, random_state=SEED),
            )
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(X_train[:, layer, :], y_train)
            path = probe_path(model_dir, layer, fold)
            path.parent.mkdir(parents=True, exist_ok=False)
            joblib.dump(pipeline, path)
            restored = joblib.load(path)
            test_features = X_test[:, layer, :]
            probabilities = restored.predict_proba(test_features)[:, 1]
            np.testing.assert_array_equal(probabilities, pipeline.predict_proba(test_features)[:, 1])
            predictions = restored.predict(test_features)
            keys = [(layer, *unpack_item(name)) for name in test_names]
            expected = original.loc[keys]
            expected_correct = expected["pass"].eq("✅").to_numpy()
            mismatches = int(np.count_nonzero((predictions == y_test) != expected_correct))
            probability_diff = float(np.max(np.abs(probabilities - expected.probability_congruent.to_numpy())))
            print(f"Fold {fold}, layer {layer}: {mismatches} prediction mismatches; "
                  f"max probability difference {probability_diff:.3g}")
    print(f"Saved {10 * len(layers)} pipelines to {output_dir}")
    return output_dir


if __name__ == "__main__":
    save_steering_probes(MODEL_ID)

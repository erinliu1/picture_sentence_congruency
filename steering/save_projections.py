"""Compute and save training-set projections onto fitted steering probes.

For every available layer and cross-validation fold, this script projects the
standardized training hidden states onto the unit logistic-regression direction.
It saves the class-specific projection distributions beside each probe and a
cross-fold summary to steering/results/<model_id>/projection_info.csv.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_vl_8b_instruct"
FOLDS = range(1, 11)


def _get_available_layers():
    """Return sorted layer indices that have steering-probe directories."""
    steering_dir = ROOT_DIR / "models" / MODEL_ID / "steering"
    layer_dirs = [path for path in steering_dir.glob("layer*") if path.is_dir()]
    if not layer_dirs:
        raise FileNotFoundError(f"No layer directories found in {steering_dir}")
    try:
        return sorted(int(path.name.removeprefix("layer")) for path in layer_dirs)
    except ValueError as error:
        raise ValueError(f"Invalid layer directory in {steering_dir}") from error


@lru_cache(maxsize=1)
def _load_fold_training_states(fold):
    """Load and cache every layer of one fold's training hidden states and labels."""
    model_dir = ROOT_DIR / "models" / MODEL_ID
    with (ROOT_DIR / "decoding/cv/cv_lookup_pt.json").open(encoding="utf-8") as file:
        cv_lookup = json.load(file)
    if str(fold) not in cv_lookup:
        raise ValueError(f"Unknown fold: {fold}")
    split = cv_lookup[str(fold)]
    train_names = [name for names in split["train"].values() for name in names]
    hidden_states, labels = [], []
    for name in train_names:
        state_path = model_dir / "hidden_states_instruction" / name
        states = torch.load(state_path, map_location="cpu", weights_only=False)
        if states.ndim != 2:
            raise ValueError(f"Expected a layer-by-feature matrix in {state_path}: {states.shape}")
        hidden_states.append(states.float().numpy())
        condition = Path(name).stem.rsplit("_", 1)[1]
        if condition not in {"congruent", "incongruent"}:
            raise ValueError(f"Unknown condition in training filename: {name}")
        labels.append(condition)
    return np.stack(hidden_states), np.asarray(labels)


def _load_training_data_and_probe(fold, layer):
    """Load one fold's raw training states, labels, scaler, and unit probe direction."""
    model_dir = ROOT_DIR / "models" / MODEL_ID
    all_hidden_states, labels = _load_fold_training_states(fold)
    if layer < 0 or layer >= all_hidden_states.shape[1]:
        raise ValueError(
            f"Layer {layer} is unavailable in fold {fold}: shape={all_hidden_states.shape}"
        )
    X = all_hidden_states[:, layer, :]
    probe_path = model_dir / "steering" / f"layer{layer}" / f"fold{fold}" / "probe.joblib"
    if not probe_path.is_file():
        raise FileNotFoundError(f"Probe not found: {probe_path}")
    pipeline = joblib.load(probe_path)

    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    w = classifier.coef_[0].copy()
    w_norm = np.linalg.norm(w)
    if not np.isfinite(w_norm) or w_norm == 0:
        raise ValueError("Probe direction must have a finite, nonzero norm")
    w_hat = w / w_norm
    return X, labels, pipeline, scaler, classifier, w_hat, w_norm


def get_training_projections(fold, layer):
    """Load or compute class-specific training projections for a fold and layer.

    Existing ``congruent_projections.npy`` and ``incongruent_projections.npy``
    files are loaded when both are present. Otherwise each raw hidden state is
    standardized with the fold's fitted scaler and projected as
    ``s_i = x_i @ w_hat``, where ``w_hat`` is the unit probe direction. The two
    returned Python lists preserve the original training-item order per class.
    """
    output_dir = (
        ROOT_DIR / "models" / MODEL_ID / "steering" / f"layer{layer}" / f"fold{fold}"
    )
    congruent_path = output_dir / "congruent_projections.npy"
    incongruent_path = output_dir / "incongruent_projections.npy"
    if congruent_path.is_file() and incongruent_path.is_file():
        congruent = np.load(congruent_path, allow_pickle=False)
        incongruent = np.load(incongruent_path, allow_pickle=False)
        if congruent.ndim != 1 or incongruent.ndim != 1:
            raise ValueError(f"Projection arrays must be one-dimensional in {output_dir}")
        if congruent.size == 0 or incongruent.size == 0:
            raise ValueError(f"Projection arrays must be nonempty in {output_dir}")
        if not np.isfinite(congruent).all() or not np.isfinite(incongruent).all():
            raise ValueError(f"Projection arrays contain nonfinite values in {output_dir}")
        return congruent.tolist(), incongruent.tolist()

    X, labels, pipeline, scaler, classifier, w_hat, w_norm = _load_training_data_and_probe(
        fold, layer
    )
    X_scaled = scaler.transform(X)
    projections = X_scaled @ w_hat
    if not np.isfinite(projections).all():
        raise ValueError(f"Nonfinite training projection in fold {fold}, layer {layer}")

    np.testing.assert_allclose(
        projections * w_norm + classifier.intercept_[0],
        pipeline.decision_function(X),
        rtol=1e-6,
        atol=1e-6,
    )
    congruent = projections[labels == "congruent"].tolist()
    incongruent = projections[labels == "incongruent"].tolist()
    if not congruent or not incongruent:
        raise ValueError(f"Both classes must be present in fold {fold}, layer {layer}")
    return congruent, incongruent


def get_training_projection_class_means(congruent_projections, incongruent_projections):
    """Return the congruent and incongruent mean training projections, respectively."""
    congruent = np.asarray(congruent_projections, dtype=np.float64)
    incongruent = np.asarray(incongruent_projections, dtype=np.float64)
    if congruent.size == 0 or incongruent.size == 0:
        raise ValueError("Both projection distributions must be nonempty")
    return float(congruent.mean()), float(incongruent.mean())


def get_training_projection_sd(congruent_projections, incongruent_projections):
    """Return s_train, the sample SD over both classes' training projections."""
    combined = np.concatenate([
        np.asarray(congruent_projections, dtype=np.float64),
        np.asarray(incongruent_projections, dtype=np.float64),
    ])
    if combined.size < 2 or not np.isfinite(combined).all():
        raise ValueError("At least two finite training projections are required")
    s_train = float(combined.std(ddof=1))
    if not np.isfinite(s_train) or s_train == 0:
        raise ValueError("Training projection SD must be finite and nonzero")
    return s_train


def get_training_standardized_separation(congruent_mean, incongruent_mean, s_train):
    """Return (congruent mean - incongruent mean) divided by s_train."""
    if not np.isfinite(s_train) or s_train <= 0:
        raise ValueError("s_train must be finite and positive")
    return float((congruent_mean - incongruent_mean) / s_train)


def save_projections(fold, layer):
    """Save both class projection arrays beside the specified fold's probe.

    Returns a dictionary containing the fold/layer projection summary used by
    ``save_all_projections``.
    """
    congruent, incongruent = get_training_projections(fold, layer)
    congruent_mean, incongruent_mean = get_training_projection_class_means(
        congruent, incongruent
    )
    s_train = get_training_projection_sd(congruent, incongruent)
    standardized_separation = get_training_standardized_separation(
        congruent_mean, incongruent_mean, s_train
    )

    output_dir = (
        ROOT_DIR / "models" / MODEL_ID / "steering" / f"layer{layer}" / f"fold{fold}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "congruent_projections.npy", np.asarray(congruent))
    np.save(output_dir / "incongruent_projections.npy", np.asarray(incongruent))
    return {
        "model_id": MODEL_ID,
        "fold": fold,
        "layer": layer,
        "s_train": s_train,
        "congruent_mean": congruent_mean,
        "incongruent_mean": incongruent_mean,
        "direction_check": "✅" if congruent_mean > incongruent_mean else "❌",
        "standardized_separation": standardized_separation,
    }


def save_all_projections():
    """Save projections for every available layer/fold and their summary CSV."""
    steering_dir = ROOT_DIR / "models" / MODEL_ID / "steering"
    layers = _get_available_layers()
    rows = []
    for fold in FOLDS:
        for layer in layers:
            fold_dir = steering_dir / f"layer{layer}" / f"fold{fold}"
            if not fold_dir.is_dir():
                raise ValueError(f"Missing fold directory: {fold_dir}")
            rows.append(save_projections(fold, layer))

    projection_info = pd.DataFrame(rows)
    results_dir = ROOT_DIR / "steering" / "results" / MODEL_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "projection_info.csv"
    projection_info.to_csv(output_path, index=False)
    print(projection_info.to_string(index=False))
    print(f"Saved {output_path}")
    return projection_info


def check_raw_hs_perturbation(k, fold, layer, verbose=True):
    """Verify one raw-hidden-state perturbation and return its delta vector.

    The raw hidden-state delta is
    ``k * s_train * scaler.scale_ * w_hat``. After standardization this must
    equal ``k * s_train * w_hat``, and its projection onto the unit direction
    must therefore change by exactly ``k * s_train``.
    """
    if not np.isfinite(k):
        raise ValueError("k must be finite")
    X, _labels, _pipeline, scaler, _classifier, w_hat, _w_norm = _load_training_data_and_probe(
        fold, layer
    )
    congruent, incongruent = get_training_projections(fold, layer)
    s_train = get_training_projection_sd(congruent, incongruent)

    # Check the algebra in float64 so adding a small delta to a large float32
    # activation does not introduce cancellation error into the assertion.
    delta = k * s_train * scaler.scale_ * w_hat
    h = X[0].astype(np.float64)
    x = scaler.transform(h[None, :])[0]
    x_steered = scaler.transform((h + delta)[None, :])[0]
    expected_scaled_delta = k * s_train * w_hat
    np.testing.assert_allclose(
        x_steered - x, expected_scaled_delta, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        (x_steered @ w_hat) - (x @ w_hat), k * s_train, rtol=1e-5, atol=1e-5
    )
    if verbose:
        print(
            f"Raw hidden-state perturbation check passed for fold {fold}, "
            f"layer {layer}, k={k:g}."
        )
    return delta


def check_raw_hs_perturbation_all(k_list):
    """Check raw-state perturbations for every fold, layer, and supplied k.

    Assertions from ``check_raw_hs_perturbation`` are allowed to propagate so
    a failed fold/layer/k combination stops the run with its numerical error.
    Returns a dataframe listing every successful combination.
    """
    k_values = [float(k) for k in k_list]
    if not k_values:
        raise ValueError("k_list must contain at least one value")
    if not np.isfinite(k_values).all():
        raise ValueError("Every k value must be finite")

    layers = _get_available_layers()
    rows = []
    for fold in FOLDS:
        for layer in layers:
            for k in k_values:
                delta = check_raw_hs_perturbation(k, fold, layer, verbose=False)
                rows.append({
                    "model_id": MODEL_ID,
                    "fold": fold,
                    "layer": layer,
                    "k": k,
                    "raw_delta_l2": float(np.linalg.norm(delta)),
                    "check": "✅",
                })
    checks = pd.DataFrame(rows)
    print(
        f"Raw hidden-state perturbation checks passed for {len(FOLDS)} folds, "
        f"{len(layers)} layers, and {len(k_values)} k values "
        f"({len(checks)} checks total)."
    )
    return checks



if __name__ == "__main__":
    save_all_projections()
    check_raw_hs_perturbation_all(k_list=[-2, -1, 0, 1, 2])

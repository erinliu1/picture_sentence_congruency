"""Fit steering sensitivity slopes from a model's steering summary."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ID = "qwen3_vl_8b_instruct"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261010


def get_true_slopes(summary_path=RESULTS_DIR / MODEL_ID / "steering_summary.csv"):
    """Return per-layer slopes and save steering_slopes.csv beside the input.

    Fit ordinary least squares with a free intercept to the five distinct
    (k, mean_delta_logit_contrast) points for each model and layer.
    """
    summary_path = Path(summary_path)
    summary = pd.read_csv(summary_path)
    rows = []
    for (model_id, layer), group in summary.groupby(["model_id", "layer"], sort=True):
        if len(group) != 5 or group.k.nunique() != 5:
            raise ValueError(f"Expected five distinct k values for {model_id}, layer {layer}")
        points = group[["k", "mean_delta_logit_contrast"]].to_numpy(dtype=float)
        if not np.isfinite(points).all():
            raise ValueError(f"Non-finite regression data for {model_id}, layer {layer}")
        slope, _intercept = np.polyfit(points[:, 0], points[:, 1], deg=1)
        rows.append({"model_id": model_id, "layer": layer, "slope": slope})

    slopes = pd.DataFrame(rows, columns=["model_id", "layer", "slope"])
    output_path = summary_path.with_name("steering_slopes.csv")
    slopes.to_csv(output_path, index=False)
    return slopes


def get_slope_bootstrap_distribution(
    summary_path=RESULTS_DIR / MODEL_ID / "steering_summary.csv",
    models_dir=MODELS_DIR,
):
    """Return 10,000 paired bootstrap slopes per layer and save their 95% CIs.

    Resample 80 stimulus sets once per replicate, sharing the draws across all
    layers and k values. Include every item in each sampled set, including
    repeated copies when a set is drawn more than once. Fit a free intercept.
    """
    summary_path = Path(summary_path)
    summary = pd.read_csv(summary_path)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, 80, size=(N_BOOTSTRAPS, 80))
    reference_sets = None
    distributions = []
    intervals = []
    for (model_id, layer), group in summary.groupby(["model_id", "layer"], sort=True):
        k_values = np.sort(group.k.to_numpy(dtype=float))
        if len(k_values) != 5 or len(np.unique(k_values)) != 5:
            raise ValueError(f"Expected five distinct k values for {model_id}, layer {layer}")
        layer_dir = Path(models_dir) / model_id / "steering" / f"layer{layer}"
        tables = []
        for fold in range(1, 11):
            fold_dir = layer_dir / f"fold{fold}"
            baseline = pd.read_csv(fold_dir / "baseline_logits.csv")
            steered = pd.read_csv(fold_dir / "steered_logits.csv")
            keys = ["item_index", "image_word", "condition"]
            merged = steered.merge(
                baseline[keys + ["logit_contrast"]], on=keys,
                how="left", validate="many_to_one", suffixes=("", "_baseline"),
            )
            merged["delta_logit_contrast"] = merged.logit_contrast - merged.logit_contrast_baseline
            tables.append(merged[merged.k.isin(k_values)])
        items = pd.concat(tables, ignore_index=True)
        if items.duplicated(keys + ["k"]).any():
            raise ValueError(f"Overlapping held-out items for {model_id}, layer {layer}")
        set_ids = np.sort(items.item_index.unique())
        if len(set_ids) != 80:
            raise ValueError(f"Expected 80 stimulus sets for {model_id}, layer {layer}")
        if reference_sets is None:
            reference_sets = set_ids
        elif not np.array_equal(reference_sets, set_ids):
            raise ValueError("Stimulus sets must match across layers for paired bootstraps")
        if not np.isfinite(items.delta_logit_contrast.to_numpy()).all():
            raise ValueError(f"Missing or non-finite deltas for {model_id}, layer {layer}")
        grouped = items.groupby(["item_index", "k"]).delta_logit_contrast
        sums = grouped.sum().unstack("k").reindex(index=set_ids, columns=k_values).to_numpy()
        counts = grouped.count().unstack("k").reindex(index=set_ids, columns=k_values).to_numpy()
        if not np.isfinite(sums).all() or not np.isfinite(counts).all() or (counts == 0).any():
            raise ValueError(f"Incomplete stimulus/k data for {model_id}, layer {layer}")
        # Sums/counts preserve item weighting even if set sizes differ.
        bootstrap_means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        centered_k = k_values - k_values.mean()
        bootstrap_slopes = bootstrap_means @ centered_k / (centered_k @ centered_k)
        distributions.append(pd.DataFrame({
            "model_id": model_id, "layer": layer,
            "bootstrap": np.arange(N_BOOTSTRAPS), "slope": bootstrap_slopes,
        }))
        lower, upper = np.quantile(bootstrap_slopes, [0.025, 0.975])
        intervals.append({
            "model_id": model_id, "layer": layer,
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
        })
    slopes = get_true_slopes(summary_path)
    slopes = slopes.merge(pd.DataFrame(intervals), on=["model_id", "layer"], validate="one_to_one")
    slopes.to_csv(summary_path.with_name("steering_slopes.csv"), index=False)
    return pd.concat(distributions, ignore_index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary_path", nargs="?", type=Path,
        default=RESULTS_DIR / MODEL_ID / "steering_summary.csv",
        help="Path to the model's steering_summary.csv",
    )
    args = parser.parse_args()
    get_slope_bootstrap_distribution(args.summary_path)
    print(pd.read_csv(args.summary_path.with_name("steering_slopes.csv")).to_string(index=False))
    print(f"Saved {args.summary_path.with_name('steering_slopes.csv')}")

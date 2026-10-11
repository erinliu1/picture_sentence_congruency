"""Plot decoding accuracy, alignment coefficient, and steering sensitivity by layer."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_vl_8b_instruct"


def plot_steering(model_id=MODEL_ID):
    """Save a shared-layer, three-panel figure as PNG and PDF; return both paths.

    Rating-readout accuracy uses instruction-conditioned decoding results.
    Steering bands show the saved 95% bootstrap slope confidence intervals.
    """
    decoding_dir = ROOT_DIR / "models" / model_id / "decoding_instruction"
    accuracy = pd.read_csv(
        ROOT_DIR / "models" / model_id / "decoding_instruction/layerwise_accuracy.csv"
    ).sort_values("layer_index")
    bootstraps = pd.read_csv(decoding_dir / "all_bootstraps.csv")
    decoding_ci = bootstraps.groupby("layer_index").all_accuracy.quantile([0.025, 0.975]).unstack()
    decoding_ci.columns = ["lower", "upper"]
    accuracy = accuracy.merge(decoding_ci, on="layer_index", how="left", validate="one_to_one")
    if not np.isfinite(accuracy[["all_accuracy", "lower", "upper"]].to_numpy(dtype=float)).all():
        raise ValueError("Decoding accuracy and bootstrap intervals must be finite")
    slopes = pd.read_csv(
        ROOT_DIR / "steering/results" / model_id / "steering_slopes.csv"
    )
    slopes = slopes[slopes.model_id == model_id].sort_values("layer")
    if accuracy.empty or slopes.empty:
        raise ValueError(f"Missing plotting data for {model_id}")
    if accuracy.layer_index.duplicated().any() or slopes.layer.duplicated().any():
        raise ValueError("Expected one observation per layer")
    if not np.array_equal(accuracy.layer_index.to_numpy(), slopes.layer.to_numpy()):
        raise ValueError("Decoding and steering layers must match")
    if not np.isfinite(slopes[["slope", "bootstrap_ci_lower", "bootstrap_ci_upper"]].to_numpy(dtype=float)).all():
        raise ValueError("Steering slopes and confidence intervals must be finite")

    alignment = pd.read_csv(
        ROOT_DIR / "models" / model_id / "alignment_instruction/coefficients.csv"
    ).sort_values("layer_index")
    if alignment.empty or alignment.layer_index.duplicated().any():
        raise ValueError("Expected one alignment observation per layer")
    if not np.array_equal(accuracy.layer_index.to_numpy(), alignment.layer_index.to_numpy()):
        raise ValueError("Decoding and alignment layers must match")
    if not np.isfinite(alignment[["coefficient", "se"]].to_numpy(dtype=float)).all():
        raise ValueError("Alignment coefficients and standard errors must be finite")
    alignment["lower"] = alignment.coefficient - 1.96 * alignment.se
    alignment["upper"] = alignment.coefficient + 1.96 * alignment.se

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(22, 5), sharex=True)
    left.plot(accuracy.layer_index, accuracy.all_accuracy, color="tab:pink", linewidth=6,
             marker="o", markersize=10, label="Rating readout decoding accuracy")
    left.fill_between(accuracy.layer_index, accuracy.lower, accuracy.upper,
                     color="tab:pink", alpha=0.2, label="95% bootstrap CI")
    left.set_title("Decoding Accuracy", fontsize=32)
    left.set_ylim(0.45, 1.0)
    left.yaxis.set_major_locator(MultipleLocator(0.1))

    middle.plot(alignment.layer_index, alignment.coefficient, color="tab:purple", linewidth=6,
                marker="o", markersize=10, label="Alignment coefficient")
    middle.fill_between(alignment.layer_index, alignment.lower, alignment.upper,
                        color="tab:purple", alpha=0.2, label="95% CI")
    middle.set_title("Alignment Coefficient", fontsize=32)
    middle.set_yticks([0, 0.5, 1, 1.5])

    right.plot(slopes.layer, slopes.slope, color="#89d3faff", linewidth=6,
                marker="o", markersize=10, label="Steering sensitivity slope")
    right.fill_between(
        slopes.layer, slopes.bootstrap_ci_lower, slopes.bootstrap_ci_upper,
        color="#89d3faff", alpha=0.2, label="95% bootstrap CI",
    )
    right.set_title("Steering Sensitivity", fontsize=32)
    right.yaxis.set_major_locator(MultipleLocator(0.5))
    right.set_xlim(slopes.layer.min(), slopes.layer.max())
    right.xaxis.set_major_locator(MaxNLocator(integer=True))
    for ax in (left, middle, right):
        ax.set_xlabel("Decoder Layer", fontsize=32)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.tick_params(axis="both", labelsize=28, width=3, length=9)
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        for spine in ax.spines.values():
            spine.set_linewidth(3)
        ax.set_axisbelow(True)
        ax.grid(True, linewidth=2.5, alpha=0.15)
    fig.tight_layout(w_pad=1.5)

    plots_dir = ROOT_DIR / "steering/plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths = tuple(plots_dir / f"{model_id}_steering.{ext}" for ext in ("png", "pdf"))
    for path in paths:
        fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    args = parser.parse_args()
    for path in plot_steering(args.model_id):
        print(f"Saved {path}")

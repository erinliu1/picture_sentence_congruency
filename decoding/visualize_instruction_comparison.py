"""Compare critical-word and rating-instruction-conditioned layerwise decoding."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/plots/gemma3_instruction")


def plot_instruction_comparison(model_id):
    with (MODELS_DIR / "models_lookup.json").open(encoding="utf-8") as file:
        title = json.load(file)[model_id]["title"]

    fig, ax = plt.subplots(figsize=(10, 5))
    for folder, label, color, linestyle in [
        ("decoding", "Without instructions", "tab:blue", "-"),
        ("decoding_instruction", "With instructions", "tab:pink", "--"),
    ]:
        results_dir = MODELS_DIR / model_id / folder
        accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
        bootstrap_df = pd.read_csv(results_dir / "all_bootstraps.csv")
        bootstrap_ci = (
            bootstrap_df.groupby("layer_index")["all_accuracy"]
            .quantile([0.025, 0.975])
            .unstack()
            .reset_index()
        )
        bootstrap_ci.columns = ["layer_index", "lower", "upper"]
        plot_df = accuracy_df.merge(bootstrap_ci, on="layer_index", validate="one_to_one")
        plot_df = plot_df.sort_values("layer_index")
        layers = plot_df["layer_index"]
        normalized_layers = (layers - layers.min()) / (layers.max() - layers.min())

        ax.plot(
            normalized_layers, plot_df["all_accuracy"],
            linewidth=4, linestyle=linestyle, color=color, label=label,
        )
        ax.fill_between(
            normalized_layers, plot_df["lower"], plot_df["upper"],
            color=color, alpha=0.2,
        )

    ax.axhline(0.5, linestyle=":", color="gray", linewidth=2, label="chance (0.50)")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Normalized layer", fontsize=20)
    ax.set_ylabel("Decoding accuracy", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.0)
    ax.tick_params(axis="both", labelsize=16, width=2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, linewidth=1.5, alpha=0.3)
    ax.legend(fontsize=15, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / f"{model_id}_instruction_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path

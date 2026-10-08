"""Plot behavioral versus final-layer decoding accuracy for instruct models."""

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import pearsonr

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent / "plots"

# Same family colors as decoding/visualize_scaling.py.
FAMILIES = {
    "qwen": ("Qwen3-VL", "#67ae00"),
    "intern": ("InternVL3", "#2563EB"),
    "plm": ("Perception-LM", "#8B5CF6"),
    "llava": ("LLaVA-OneVision", "#4fd6ff"),
    "glm": ("GLM-4.6V", "#EC4899"),
    "gemma3": ("Gemma3", "#ffd42b"),
    "gemma4": ("Gemma4", "#DC2626"),
}
LABEL_FS = 24
TICK_FS = 22
LEGEND_FS = 20


def load_pairs(folder, models_lookup, behavioral):
    rows = []
    for model_id, info in models_lookup.items():
        if info["family"].endswith("_base"):
            continue
        path = ROOT_DIR / "models" / model_id / folder / "layerwise_accuracy.csv"
        if model_id not in behavioral.index or not path.exists():
            warnings.warn(f"Missing {folder} inputs for {model_id}", stacklevel=2)
            continue
        accuracy = pd.read_csv(path).sort_values("layer_index")
        if accuracy.empty or accuracy["layer_index"].isna().any() or accuracy["layer_index"].duplicated().any():
            raise ValueError(f"Empty or invalid layer indices: {path}")
        rows.append({
            "model_id": model_id,
            "family": info["family"],
            "behavioral_accuracy": behavioral.loc[model_id],
            "final_layer_accuracy": accuracy["all_accuracy"].iloc[-1],
        })
    return pd.DataFrame(rows).dropna(subset=["behavioral_accuracy", "final_layer_accuracy"])


def visualize_behavioral_decoding_correlation():
    with (ROOT_DIR / "models/models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)
    behavioral = pd.read_csv(ROOT_DIR / "behavioral/results/accuracy.csv")
    if behavioral["source"].duplicated().any():
        raise ValueError("Behavioral accuracy contains duplicate model IDs")
    behavioral = behavioral.set_index("source")["accuracy"]
    # Reserve space below a roughly 10 x 5 plotting area for the shared legend.
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.4), sharex=True, sharey=True)
    plotted_families = set()
    for ax, folder, title in zip(
        axes, ["decoding", "decoding_instruction"], ["Critical word", "Rating readout"]
    ):
        pairs = load_pairs(folder, models_lookup, behavioral)
        if len(pairs) < 3 or (pairs[["final_layer_accuracy", "behavioral_accuracy"]].nunique() < 2).any():
            raise ValueError(f"Insufficient or constant correlation inputs: {folder}")
        for family, group in pairs.groupby("family"):
            _, color = FAMILIES[family]
            plotted_families.add(family)
            ax.scatter(group["final_layer_accuracy"], group["behavioral_accuracy"],
                       color=color, s=130, alpha=0.7, edgecolors="white", linewidths=0.7, zorder=3)
        x = pairs["final_layer_accuracy"]
        y = pairs["behavioral_accuracy"]
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        ax.plot(line_x, slope * line_x + intercept, color="black", linewidth=2, zorder=2)
        r = pearsonr(x, y).statistic
        ax.text(0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=LABEL_FS,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                      "edgecolor": "lightgray", "linewidth": 1, "alpha": 0.9})
        ax.set_title(title, fontsize=LABEL_FS, pad=10)
        ax.set_xlim(0.65, 1.0)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0], ["0.7", "0.8", "0.9", "1.0"])
        ax.tick_params(axis="both", labelsize=TICK_FS, width=1.5, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, linewidth=1, alpha=0.3)
        print(f"{title}: n={len(pairs)}, Pearson r={r:.6f}")
    axes[0].set_ylabel("Behavioral accuracy", fontsize=LABEL_FS)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=color, markeredgecolor="white", label=label)
        for family, (label, color) in FAMILIES.items() if family in plotted_families
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=LEGEND_FS,
               bbox_to_anchor=(0.5, 0.005), framealpha=0.9, columnspacing=0.7,
               handletextpad=0.4)
    fig.supxlabel("Final-layer decoding accuracy", fontsize=LABEL_FS, y=0.205)
    fig.subplots_adjust(left=0.115, right=0.98, top=0.91, bottom=0.36, wspace=0.12)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "behavioral_decoding_correlation.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    visualize_behavioral_decoding_correlation()

"""Plot behavioral versus final-layer decoding accuracy for instruct models."""

import json
import re
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


TYPE_MARKERS = {"dense": "o", "moe": "D", "PLE": "^"}
RED_OUTLINE = {"llava_ov_05b", "plm_1b", "plm_8b"}
BLUE_OUTLINE = {"plm_1b", "llava_ov_05b", "gemma4_e2b", "internvl3_1b_instruct"}


def marker_size(size):
    """Same total/effective parameter-count parsing and log-area scale as behavioral plot."""
    params = float(re.search(r"\d+(?:\.\d+)?", size).group())
    fraction = np.clip((np.log10(params) - np.log10(0.4)) / (np.log10(250) - np.log10(0.4)), 0, 1)
    return 90 + fraction * (700 - 90)


def draw_marker(ax, row, info):
    color = FAMILIES[row["family"]][1]
    marker = TYPE_MARKERS[info["type"]]
    size = marker_size(info["size"])
    red = row["model_id"] in RED_OUTLINE
    blue = row["model_id"] in BLUE_OUTLINE
    edge = "none" if red and blue else "red" if red else "#1200ff" if blue else "black"
    ax.scatter(row["final_layer_accuracy"], row["behavioral_accuracy"],
               color=color, marker=marker, s=size, alpha=0.6, edgecolors=edge,
               linewidths=2 if red or blue else 1,
               linestyles="dotted" if red or blue else "solid", zorder=3)
    if red and blue:
        for color, offset in [("red", 0), ("#1200ff", 2)]:
            ax.scatter(row["final_layer_accuracy"], row["behavioral_accuracy"],
                       facecolors="none", marker=marker, s=size, edgecolors=color,
                       linewidths=2, linestyles=(offset, (1, 3)), zorder=4)


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
            "decoding": "critical_word" if folder == "decoding" else "rating_readout",
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
        for _, row in pairs.iterrows():
            plotted_families.add(row["family"])
            draw_marker(ax, row, models_lookup[row["model_id"]])
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
        ax.set_xlim(0.65, 1.025)
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
    out_path = PLOTS_DIR / "behavioral_decoding_correlation_styled.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    # Separate architecture key preserves the compact shared family legend.
    key_fig = plt.figure(figsize=(6, 1))
    key_handles = [Line2D([0], [0], marker=marker, linestyle="none",
                         markerfacecolor="gray", markeredgecolor="black",
                         markersize=10, label=label)
                   for marker, label in [("o", "Dense"), ("D", "MoE"), ("^", "PLE")]]
    key_fig.legend(handles=key_handles, loc="center", ncol=3, fontsize=18)
    key_fig.savefig(PLOTS_DIR / "behavioral_decoding_architecture_legend.png", dpi=150)
    plt.close(key_fig)
    out_path.with_suffix(".txt").write_text(
        "Instruction-tuned VLMs; each point represents one model. "
        "Color denotes family; circles denote dense models, diamonds MoE, triangles PLE. "
        "Marker area uses the behavioral plot's log-scaled total parameter count "
        "(effective count for E2B/E4B). Red dotted outlines: LLaVA-OV-0.5B, PLM-1B, PLM-8B. "
        "Blue dotted outlines: PLM-1B, LLaVA-OV-0.5B, Gemma4-E2B, InternVL3-1B. "
        "Models in both lists have alternating red/blue outlines. The black line is ordinary "
        "least-squares regression across models; r is the Pearson correlation of point estimates.\n",
        encoding="utf-8")
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    visualize_behavioral_decoding_correlation()

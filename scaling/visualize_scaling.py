"""Plot peak CW/RR decoding and alignment versus log10 parameter count.

OLS fits and Pearson correlations use log10(params_b), across instruct models.
Marker encodings match the styled behavioral/decoding correlation figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext
from scipy.stats import pearsonr

ROOT_DIR = Path(__file__).resolve().parents[1]
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FAMILIES = {
    "qwen": ("Qwen3-VL", "#67ae00"),
    "intern": ("InternVL3", "#2563EB"),
    "plm": ("Perception-LM", "#8B5CF6"),
    "llava": ("LLaVA-OneVision", "#4fd6ff"),
    "glm": ("GLM-4.6V", "#EC4899"),
    "gemma3": ("Gemma3", "#ffd42b"),
    "gemma4": ("Gemma4", "#DC2626"),
}
TYPE_MARKERS = {"dense": "o", "moe": "D", "PLE": "^"}
LABEL_FS = 24
TICK_FS = 20


def draw_marker(ax, row, metric, info):
    size = 130
    marker = TYPE_MARKERS[info["type"]]
    ax.scatter(
        row.params_b, getattr(row, metric), color=FAMILIES[row.model_family][1],
        marker=marker, s=size, alpha=0.6, edgecolors="black",
        linewidths=1, linestyles="solid", zorder=3,
    )


def visualize_scaling():
    summary = pd.read_csv(Path(__file__).resolve().parent / "results/scaling_summary.csv")
    with (ROOT_DIR / "models/models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)
    instruct_ids = {mid for mid, info in models_lookup.items() if not info["family"].endswith("_base")}
    if summary.model_id.duplicated().any() or set(summary.model_id) != instruct_ids:
        raise ValueError("Run scaling_summarize.py: expected exactly one row per instruct model")
    metrics = ["peak_CW_decoding", "peak_RR_decoding", "peak_CW_alignment", "peak_RR_alignment"]
    if summary[["params_b", "log_parameters", *metrics]].isna().any().any():
        raise ValueError("Run scaling_summarize.py: incomplete scaling metrics")
    if not np.allclose(summary.log_parameters, np.log10(summary.params_b)):
        raise ValueError("log_parameters must equal log10(params_b)")

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey="row")
    for ax, metric in zip(axes.flat, metrics):
        for row in summary.itertuples(index=False):
            draw_marker(ax, row, metric, models_lookup[row.model_id])
        x, y = summary.log_parameters, summary[metric]
        slope, intercept = np.polyfit(x, y, 1)
        log_grid = np.linspace(x.min(), x.max(), 200)
        ax.plot(10 ** log_grid, slope * log_grid + intercept, color="black", linewidth=2, zorder=2)
        r = pearsonr(x, y).statistic
        ax.text(
            0.95, 0.05, f"r = {r:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=LABEL_FS,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "lightgray", "linewidth": 1, "alpha": 0.9},
        )
        ax.set_xscale("log")
        ax.set_xlim(0.08, 1500)
        ax.set_xticks([0.1, 1, 10, 100, 1000])
        ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
        row_index, col_index = np.argwhere(axes == ax)[0]
        ax.tick_params(axis="both", labelsize=TICK_FS, width=1.5, length=6,
                       labelbottom=(row_index == 1), labelleft=(col_index == 0))
        ax.tick_params(axis="x", which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, linewidth=1, alpha=0.3)
        print(f"{metric}: n={len(summary)}, Pearson r={r:.6f}")

    for col, title in enumerate(["Critical word", "Rating readout"]):
        axes[0, col].set_title(title, fontsize=LABEL_FS, pad=14)
    axes[0, 0].set_ylabel("Peak decoding\naccuracy", fontsize=LABEL_FS)
    axes[1, 0].set_ylabel("Peak alignment\ncoefficient", fontsize=LABEL_FS)
    fig.supxlabel("Model Size (Billions of Parameters)", fontsize=LABEL_FS, y=0.175)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
               markerfacecolor=color, markeredgecolor="white", label=label)
        for label, color in FAMILIES.values()
    ]
    handles.extend(
        Line2D([0], [0], marker=marker, linestyle="none", markersize=9,
               markerfacecolor="gray", markeredgecolor="black", label=label)
        for marker, label in [("o", "Dense"), ("D", "MoE"), ("^", "PLE")]
    )
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=4, fontsize=18, framealpha=0.9, columnspacing=0.8, handletextpad=0.4)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.94, bottom=0.26, hspace=0.18, wspace=0.12)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / "scaling.pdf"
    fig.savefig(out_path)
    fig.savefig(out_path.with_suffix(".png"), dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    visualize_scaling()

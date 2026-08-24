# One scatter plot of every model's minimum instruction-following probability mass (x) against its
# image-grounding correlation (y).
#
# Encodings (same convention as visualize_metrics_summary.py):
#   - color     -> model family (qwen / intern / llava / plm / glm / gemma3 / gemma4)
#   - marker    -> model type (dense = circle, MoE = diamond, PLE = triangle)
#   - marker size -> approximate parameter count (log-scaled; capped smaller than metrics_summary's)
#   - dashed red outline   -> instruction-following flag: min_total_probability_mass < 0.90
#   - dashed neon-green outline -> high grounding correlation: grounding_r > 0.20
#   - alternating red/neon-green dashed outline -> both of the above

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
BEHAVIORAL_RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
PLOTS_DIR.mkdir(exist_ok=True)

with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

# min_total_probability_mass lives in instruction_following.csv (health.csv only holds rating/
# top-one-token stats, not probability-mass data).
instruction_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "instruction_following.csv").set_index("source")
grounding_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "grounding.csv").set_index("source")

FAMILY_LABEL = {
    "qwen": "Qwen3-VL", "intern": "InternVL3", "llava": "LLaVA-OV", "plm": "PLM",
    "glm": "GLM-4.6V", "gemma3": "Gemma 3", "gemma4": "Gemma 4",
}

family_colors = {
    "qwen": "#67ae00",     # green
    "intern": "#2563EB",   # blue
    "plm": "#8B5CF6",      # violet
    "llava": "#4fd6ff",    # pink/magenta
    "glm": "#EC4899",      # cyan
    "gemma3": "#ffd42b",   # amber
    "gemma4": "#DC2626",   # red-orange
}

type_marker = {"dense": "o", "moe": "D", "PLE": "^"}
type_label = {"dense": "Dense", "moe": "MoE", "PLE": "PLE"}


def get_plot_df():
    rows = []
    for model_id, info in models_lookup.items():
        if model_id not in instruction_df.index or model_id not in grounding_df.index:
            continue
        min_mass = instruction_df.loc[model_id, "min_total_probability_mass"]
        rows.append({
            "model_id": model_id,
            "family": info["family"],
            "type": info["type"],
            # Leading number in the size string, in billions of parameters (e.g. "30B-A3B" -> 30, "106-A12B" -> 106, "E2B" -> 2).
            "params_b": float(re.search(r"(\d+(?:\.\d+)?)", info["size"]).group(1)),
            "min_mass": min_mass,
            "grounding_r": grounding_df.loc[model_id, "grounding_r"],
            "instruction_flagged": min_mass < 0.90,
        })
    return pd.DataFrame(rows)


def marker_size(params_b, s_min=90, s_max=700):
    """Log-scaled scatter marker area, from smallest to largest model (capped smaller than
    visualize_metrics_summary.py's, since here every model shares one crowded panel)."""
    log_p = np.log10(params_b)
    log_min, log_max = np.log10(0.4), np.log10(250)
    t = np.clip((log_p - log_min) / (log_max - log_min), 0, 1)
    return s_min + t * (s_max - s_min)


plot_df = get_plot_df()

fig, ax = plt.subplots(figsize=(5, 5))

DASH = 3  # dash segment length (points) shared by the red/green alternating outline below

for _, row in plot_df.iterrows():
    color = family_colors[row["family"]]
    marker = type_marker[row["type"]]
    size = marker_size(row["params_b"])
    low_mass = row["instruction_flagged"]  # min_total_probability_mass < 0.90
    high_grounding = row["grounding_r"] > 0.20

    if low_mass and high_grounding:
        # both flagged: alternate red/green dashes by overlaying two complementary dash patterns
        edge_kwargs = dict(edgecolors="none")
    elif high_grounding:
        edge_kwargs = dict(edgecolors="#1200ff", linewidths=2, linestyles="dashed")
    elif low_mass:
        edge_kwargs = dict(edgecolors="red", linewidths=2, linestyles="dashed")
    else:
        edge_kwargs = dict(edgecolors="black", linewidths=1, linestyles="solid")

    ax.scatter(row["min_mass"], row["grounding_r"], color=color, marker=marker, s=size, alpha=0.6, zorder=3, **edge_kwargs)

    if low_mass and high_grounding:
        for dash_color, offset in [("red", 0), ("#1200ff", DASH)]:
            ax.scatter(
                row["min_mass"], row["grounding_r"], facecolors="none", marker=marker, s=size,
                edgecolors=dash_color, linewidths=2, linestyles=(offset, (DASH, DASH)), zorder=4,
            )

# ax.axvline(0.90, color="grey", linestyle="--", linewidth=2, zorder=1, alpha=0.6)
# ax.axhline(0.2, color="grey", linestyle="--", linewidth=2, zorder=1, alpha=0.6)

# ax.text(
#     0.885, plot_df["grounding_r"].min() + 0.02,
#     "Instruction-following threshold (0.90)",
#     rotation=90, va="bottom", ha="right",
#     fontsize=14, color="grey", zorder=1,
# )

ax.tick_params(axis="both", labelsize=22, width=2, length=8)
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.grid(True, linewidth=1, alpha=0.3)

ax.set_xlabel("Minimum Rating Mass", fontsize=22)
ax.set_ylabel("Grounding Correlation", fontsize=22)

# type_handles = [
#     Line2D(
#         [0], [0],
#         marker=marker,
#         linestyle="None",
#         markerfacecolor="none",
#         markeredgecolor="black",
#         markeredgewidth=2,
#         markersize=12,
#         label=type_label[t],
#     )
#     for t, marker in type_marker.items()
# ]

# ax.legend(
#     handles=type_handles,
#     loc="lower left",
#     fontsize=14,
#     framealpha=0.9,
# )

# family_handles = [
#     Line2D(
#         [0], [0],
#         marker="o",
#         linestyle="None",
#         color=family_colors[family],
#         markersize=12,
#         label=label,
#     )
#     for family, label in FAMILY_LABEL.items()
# ]

# fig.legend(
#     handles=family_handles,
#     loc="lower center",
#     bbox_to_anchor=(0.5, -0.02),
#     ncol=4,
#     fontsize=14,
#     framealpha=0.9,
# )

fig.subplots_adjust(left=0.15, right=0.97, top=0.97, bottom=0.23)

plt.savefig(PLOTS_DIR / "instruction_grounding.png", bbox_inches="tight", dpi=600)
plt.show()


def save_standalone_legend(handles, ncol, out_path, fontsize=18, height=None):
    """Render just a legend (no axes/data) to its own tightly-cropped PNG.
    If `height` is given (inches), the saved PNG is exactly that tall (legend vertically
    centered within it) rather than cropped tight to the legend's own extent."""
    fig_legend = plt.figure(figsize=(0.1, 0.1))
    legend = fig_legend.legend(handles=handles, loc="center", ncol=ncol, fontsize=fontsize, framealpha=0)
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    if height is not None:
        y_center = (bbox.y0 + bbox.y1) / 2
        bbox = Bbox.from_extents(bbox.x0, y_center - height / 2, bbox.x1, y_center + height / 2)
    fig_legend.savefig(out_path, dpi=600, bbox_inches=bbox)
    plt.close(fig_legend)


type_handles = [
    Line2D(
        [0], [0],
        marker=marker,
        linestyle="None",
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=2,
        markersize=12,
        label=type_label[t],
    )
    for t, marker in type_marker.items()
]

family_handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="None",
        color=family_colors[family],
        markersize=12,
        label=label,
    )
    for family, label in FAMILY_LABEL.items()
]

# family colors: 1 column x 7 rows, height 5in, fontsize 20
save_standalone_legend(family_handles, ncol=1, out_path=PLOTS_DIR / "instruction_grounding_family_legend.png", fontsize=20, height=5)
# marker type: 3 rows x 1 column
save_standalone_legend(type_handles, ncol=1, out_path=PLOTS_DIR / "instruction_grounding_type_legend.png", fontsize=20)

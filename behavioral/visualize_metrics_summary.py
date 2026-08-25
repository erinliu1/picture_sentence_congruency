# One two-panel figure comparing every model's classification accuracy (left) and Spearman's rho vs. human ratings (right), on a shared, rho-sorted y-axis.
#
# Encodings (same convention as visualize_instruction_grounding.py):
#   - color     -> model family (qwen / intern / llava / plm / glm / gemma3 / gemma4)
#   - marker    -> model type (dense = circle, MoE = diamond, PLE = triangle)
#   - marker size -> approximate parameter count (log-scaled; total params for MoE, "effective" params for PLE)
#   - dashed red outline   -> instruction-following flag: total_probability_mass < 0.90 on at least one trial
#   - dashed blue outline  -> high grounding correlation: grounding_r > 0.20
#   - alternating red/blue dashed outline -> both of the above

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
BEHAVIORAL_RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
PLOTS_DIR.mkdir(exist_ok=True)

with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

accuracy_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "accuracy.csv").set_index("source")
human_accuracy = accuracy_df.loc["human", "accuracy"]

correlation_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "correlation.csv").set_index("source")
grounding_df = pd.read_csv(BEHAVIORAL_RESULTS_DIR / "grounding.csv").set_index("source")

FAMILY_LABEL = {
    "qwen": "Qwen3-VL", "intern": "InternVL3", "llava": "LLaVA-OneVision", "plm": "PLM",
    "glm": "GLM-4.6V", "gemma3": "Gemma 3", "gemma4": "Gemma 4",
}

# Short per-family stems used for the row labels (as opposed to FAMILY_LABEL, used in the legend).
FAMILY_SHORT = {
    "qwen": "Qwen3-VL", "intern": "InternVL3", "llava": "LLaVA-OV", "plm": "PLM",
    "glm": "GLM", "gemma3": "Gemma3", "gemma4": "Gemma4",
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
        ratings_df = pd.read_csv(MODELS_DIR / model_id / "behavior" / "ratings.csv")

        rows.append({
            "model_id": model_id,
            "title": info["title"],
            # Compact row label: family stem + size in parentheses, dropping "-Instruct" (e.g. "Qwen3-VL (30B-A3B)").
            "short_title": f"{FAMILY_SHORT[info['family']]} ({info['size']})",
            "family": info["family"],
            "type": info["type"],
            "size": info["size"],
            # Leading number in the size string, in billions of parameters (e.g. "30B-A3B" -> 30, "106-A12B" -> 106, "E2B" -> 2). Used as the log x-axis position.
            "params_b": float(re.search(r"(\d+(?:\.\d+)?)", info["size"]).group(1)),
            "accuracy": accuracy_df.loc[model_id, "accuracy"],
            "rho": correlation_df.loc[model_id, "spearman_rho"],
            "ci_lower": correlation_df.loc[model_id, "ci_lower"],
            "ci_upper": correlation_df.loc[model_id, "ci_upper"],
            # flag if any trial's total_probability_mass < 0.90, indicating instruction-following degradation
            "instruction_flagged": ratings_df["total_probability_mass"].min() < 0.90,
            "grounding_r": grounding_df.loc[model_id, "grounding_r"],
            "high_grounding": grounding_df.loc[model_id, "grounding_r"] > 0.20,
        })
    df = pd.DataFrame(rows).sort_values("rho", ascending=True).reset_index(drop=True)
    return df


def marker_size(params_b, s_min=90, s_max=1600):
    """Log-scaled scatter marker area, from smallest to largest model."""
    log_p = np.log10(params_b)
    log_min, log_max = np.log10(0.4), np.log10(250)
    t = np.clip((log_p - log_min) / (log_max - log_min), 0, 1)
    return s_min + t * (s_max - s_min)


plot_df = get_plot_df()
y_pos = np.arange(len(plot_df))

fig, (ax_acc, ax_rho) = plt.subplots(1, 2, figsize=(10, 12.5), sharey=True, gridspec_kw={"wspace": 0.4})

DASH = 3  # dash segment length (points) shared by the red/blue alternating outline below

for i, row in plot_df.iterrows():
    color = family_colors[row["family"]]
    marker = type_marker[row["type"]]
    size = marker_size(row["params_b"])
    low_mass = row["instruction_flagged"]  # total_probability_mass < 0.90 on at least one trial
    high_grounding = row["high_grounding"]  # grounding_r > 0.20

    if low_mass and high_grounding:
        # both flagged: alternate red/blue dashes by overlaying two complementary dash patterns
        edge_kwargs = dict(edgecolors="none")
    elif high_grounding:
        edge_kwargs = dict(edgecolors="#1200ff", linewidths=2, linestyles="dashed")
    elif low_mass:
        edge_kwargs = dict(edgecolors="red", linewidths=2, linestyles="dashed")
    else:
        edge_kwargs = dict(edgecolors="black", linewidths=1, linestyles="solid")

    ax_acc.scatter(row["accuracy"], i, color=color, marker=marker, s=size, alpha=0.6, zorder=3, **edge_kwargs)
    ax_acc.scatter(row["accuracy"], i, color="black", marker="o", s=12, alpha=1, zorder=4)

    ax_rho.errorbar(
        row["rho"], i,
        xerr=[[row["rho"] - row["ci_lower"]], [row["ci_upper"] - row["rho"]]],
        fmt="none", ecolor="black", elinewidth=2, capsize=3, zorder=10,
    )
    ax_rho.scatter(row["rho"], i, color=color, marker=marker, s=size, alpha=0.6, zorder=3, **edge_kwargs)
    ax_rho.scatter(row["rho"], i, color="black", marker="o", s=12, alpha=1, zorder=11)

    if low_mass and high_grounding:
        for dash_color, offset in [("red", 0), ("#1200ff", DASH)]:
            ax_acc.scatter(
                row["accuracy"], i, facecolors="none", marker=marker, s=size,
                edgecolors=dash_color, linewidths=2, linestyles=(offset, (DASH, DASH)), zorder=5,
            )
            ax_rho.scatter(
                row["rho"], i, facecolors="none", marker=marker, s=size,
                edgecolors=dash_color, linewidths=2, linestyles=(offset, (DASH, DASH)), zorder=12,
            )

ax_acc.axvline(human_accuracy, color="grey", linestyle="--", linewidth=2, zorder=1)
ax_acc.text(
    human_accuracy - 0.015, 3.5,
    f"Human accuracy ({human_accuracy*100:.1f}%)",
    rotation=90, va="center", ha="right",
    fontsize=18, color="grey", zorder=1,
)

ax_acc.set_yticks(y_pos)
ax_acc.set_yticklabels(plot_df["short_title"])
ax_acc.set_ylim(-1, len(plot_df))

for ax in (ax_acc, ax_rho):
    ax.tick_params(axis="both", which="major", labelsize=22, width=2, length=8)
    ax.tick_params(axis="x", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, which="major", linewidth=1, alpha=0.3)
    ax.grid(True, which="minor", axis="x", linewidth=1, alpha=0.3)

ax_acc.tick_params(axis="y", labelsize=16)
ax_rho.tick_params(labelleft=False)

ax_acc.set_xticks([0.5, 1])
ax_acc.set_xticks([0.5, 0.75, 1], minor=True)
ax_rho.set_xticks([0, 1])
ax_rho.set_xticks([0, 0.25, 0.5, 0.75, 1], minor=True)

ax_acc.set_xlabel("Accuracy", fontsize=22, labelpad=-22)
ax_rho.set_xlabel("Spearman's ρ", fontsize=22, labelpad=-22)

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

ax_acc.legend(
    handles=type_handles,
    loc="upper left",
    fontsize=14,
    title_fontsize=18,
    framealpha=0.9,
)

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

fig.legend(
    handles=family_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=4,
    fontsize=18,
    framealpha=0.9,
)

fig.subplots_adjust(left=0.2, right=0.97, top=0.97, bottom=0.11)

plt.savefig(PLOTS_DIR / "metrics_summary.pdf", bbox_inches="tight")
plt.show()


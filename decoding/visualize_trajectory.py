import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


def clean_01_formatter(value, pos):
    """Format tick labels normally, but show 0 and 1 without a trailing decimal."""
    if np.isclose(value, 0):
        return "0"
    if np.isclose(value, 1):
        return "1"
    return f"{value:g}"

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/plots")

N_POINTS = 101
common_grid = np.linspace(0, 1, N_POINTS)

decoding_color = "tab:green"   # green
alignment_color = "tab:purple"  # purple


def normalized_layers(layer_index):
    return (layer_index - layer_index.min()) / (layer_index.max() - layer_index.min())


def mean_and_ci(stacked):
    """stacked: (n_models, N_POINTS) array. Returns mean, lower, upper (95% CI of the mean across models)."""
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


decoding_curves = []
alignment_curves = []
significance_thresholds = []

for model_id in models_lookup:
    model_dir = MODELS_DIR / model_id

    accuracy_df = pd.read_csv(model_dir / "decoding" / "layerwise_accuracy.csv")
    depth = normalized_layers(accuracy_df["layer_index"])
    decoding_curves.append(np.interp(common_grid, depth, accuracy_df["all_accuracy"]))

    significance_df = pd.read_csv(model_dir / "decoding" / "layerwise_significance.csv")
    significance_thresholds.append(significance_df["threshold_95"].iloc[0])

    coefficients_df = pd.read_csv(model_dir / "alignment" / "coefficients.csv")
    depth = normalized_layers(coefficients_df["layer_index"])
    # Normalize each model's own trajectory by its own peak, so every model
    # contributes on the same 0-1-ish "fraction of its max alignment" scale.
    peak_normalized_coeff = coefficients_df["coefficient"] / coefficients_df["coefficient"].max()
    alignment_curves.append(np.interp(common_grid, depth, peak_normalized_coeff))

decoding_mean, decoding_lower, decoding_upper = mean_and_ci(np.stack(decoding_curves))
alignment_mean, alignment_lower, alignment_upper = mean_and_ci(np.stack(alignment_curves))
average_significance_threshold = np.mean(significance_thresholds)

LABEL_FS = 22
TICK_FS = 18

fig, (ax_dec, ax_align) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.55})

ax_dec.plot(common_grid, decoding_mean, color=decoding_color, linewidth=5, zorder=2)
ax_dec.fill_between(common_grid, decoding_lower, decoding_upper, color=decoding_color, alpha=0.2)
ax_dec.axhline(
    average_significance_threshold,
    linestyle="--",
    color="gray",
    linewidth=3,
    zorder=1,
)
ax_dec.text(
    0.07,
    average_significance_threshold - 0.035,
    f"average significance threshold",
    color="gray",
    fontsize=14,
    # ha="left",
    # va="bottom",
)

ax_align.plot(common_grid, alignment_mean, color=alignment_color, linewidth=4, zorder=2)
ax_align.fill_between(common_grid, alignment_lower, alignment_upper, color=alignment_color, alpha=0.2)

for ax in (ax_dec, ax_align):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel("Relative Layer Depth", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.5, length=6)
    ax.xaxis.set_major_formatter(FuncFormatter(clean_01_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_01_formatter))
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linewidth=1, alpha=0.3)

ax_dec.set_ylim(0.5, 1.0)
ax_dec.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax_align.set_ylim(0, 1)
ax_align.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])

ax_dec.set_ylabel("Decoding Accuracy", fontsize=LABEL_FS)
ax_align.set_ylabel("Alignment Coefficient\n(fraction of peak)", fontsize=LABEL_FS)

ax_dec.set_title("Decoding Trajectory", fontsize=LABEL_FS, pad=10)
ax_align.set_title("Alignment Trajectory", fontsize=LABEL_FS, pad=10)

fig.savefig(PLOTS_DIR / "trajectory.pdf", bbox_inches="tight")
plt.show()

caption = (
    "Average decoding accuracy trajectory and average alignment coefficient trajectory "
    "relative to model peak, across normalized network depth (28 models). Left: the "
    "layerwise congruent-vs-incongruent decoding accuracy of each model was interpolated "
    "onto 101 evenly spaced normalized depth positions (0 = first hidden-state layer, "
    "1 = last), then averaged across all 28 models; the shaded band is the 95% CI of the "
    "across-model mean at each depth. The dotted gray line marks the average permutation-"
    "based 95th-percentile significance threshold (from layerwise_significance.csv), "
    "averaged across all 28 models. Right: each model's layerwise alignment coefficient "
    "(the OLS regression coefficient of human congruency ratings on the model's decoded "
    "congruent-probability) was first divided by that model's own peak coefficient, so "
    "every model's trajectory has a peak of 1 regardless of its absolute alignment "
    "strength; these peak-normalized trajectories were then interpolated onto the same "
    "101-point normalized-depth grid and averaged across all 28 models, with the shaded "
    "band showing the 95% CI of the across-model mean. Together the two panels show the "
    "common layerwise shape of decodability and human-alignment across architecturally "
    "and scale-diverse vision-language models."
)
with open(PLOTS_DIR / "trajectory_caption.txt", "w", encoding="utf-8") as file:
    file.write(caption + "\n")

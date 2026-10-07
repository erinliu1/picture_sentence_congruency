import json
import warnings
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

decoding_color = "tab:green"   # instruction-tuned, without instructions
instruction_color = "tab:pink"
decoding_base_color = "tab:orange"  # orange (base / pretrained, non-instruction-tuned)
alignment_color = "tab:green"  # instruction-tuned, with instructions

# Base (pretrained, non-instruction-tuned) counterparts don't go through the alignment stage
# (no human-rating regression has been run for them), so they only contribute a second decoding
# curve here, not an alignment curve.
BASE_FAMILIES = {"gemma3_base", "gemma4_base", "intern_base"}


def normalized_layers(layer_index):
    return (layer_index - layer_index.min()) / (layer_index.max() - layer_index.min())


def mean_and_ci(stacked):
    """stacked: (n_models, N_POINTS) array. Returns mean, lower, upper (95% CI of the mean across models)."""
    mean = stacked.mean(axis=0)
    sem = stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
    return mean, mean - 1.96 * sem, mean + 1.96 * sem


decoding_curves = []
decoding_instruction_curves = []
decoding_base_curves = []
alignment_curves = []
alignment_instruction_curves = []
significance_thresholds = []

for model_id, info in models_lookup.items():
    model_dir = MODELS_DIR / model_id
    is_base = info["family"] in BASE_FAMILIES

    accuracy_df = pd.read_csv(model_dir / "decoding" / "layerwise_accuracy.csv").sort_values("layer_index")
    depth = normalized_layers(accuracy_df["layer_index"])
    interpolated_accuracy = np.interp(common_grid, depth, accuracy_df["all_accuracy"])

    if is_base:
        # Base models don't have permutation-significance or alignment stages run -- only
        # decoding -- so they contribute a decoding curve only.
        decoding_base_curves.append(interpolated_accuracy)
        continue

    decoding_curves.append(interpolated_accuracy)

    significance_df = pd.read_csv(model_dir / "decoding" / "layerwise_significance.csv")
    significance_thresholds.append(significance_df["threshold_95"].iloc[0])

    coefficients_df = pd.read_csv(model_dir / "alignment" / "coefficients.csv").sort_values("layer_index")
    depth = normalized_layers(coefficients_df["layer_index"])
    # Normalize each model's own trajectory by its own peak, so every model
    # contributes on the same 0-1-ish "fraction of its max alignment" scale.
    peak_normalized_coeff = coefficients_df["coefficient"] / coefficients_df["coefficient"].max()
    alignment_curves.append(np.interp(common_grid, depth, peak_normalized_coeff))

    for folder, filename, value_column, curves in [
        ("decoding_instruction", "layerwise_accuracy.csv", "all_accuracy", decoding_instruction_curves),
        ("alignment_instruction", "coefficients.csv", "coefficient", alignment_instruction_curves),
    ]:
        result_path = model_dir / folder / filename
        if not result_path.exists():
            warnings.warn(f"Skipping missing instruction-conditioned results: {result_path}")
            continue
        result_df = pd.read_csv(result_path).sort_values("layer_index")
        depth = normalized_layers(result_df["layer_index"])
        values = result_df[value_column]
        if value_column == "coefficient":
            values = values / values.max()
        curves.append(np.interp(common_grid, depth, values))

decoding_mean, decoding_lower, decoding_upper = mean_and_ci(np.stack(decoding_curves))
decoding_instruction_mean, decoding_instruction_lower, decoding_instruction_upper = mean_and_ci(np.stack(decoding_instruction_curves))
decoding_base_mean, decoding_base_lower, decoding_base_upper = mean_and_ci(np.stack(decoding_base_curves))
alignment_mean, alignment_lower, alignment_upper = mean_and_ci(np.stack(alignment_curves))
alignment_instruction_mean, alignment_instruction_lower, alignment_instruction_upper = mean_and_ci(np.stack(alignment_instruction_curves))
average_significance_threshold = np.mean(significance_thresholds)

LABEL_FS = 22
TICK_FS = 18

fig, (ax_dec, ax_align) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.55})

ax_dec.plot(common_grid, decoding_mean, color=decoding_color, linewidth=5, zorder=2)
ax_dec.fill_between(common_grid, decoding_lower, decoding_upper, color=decoding_color, alpha=0.2)
ax_dec.plot(common_grid, decoding_instruction_mean, color=instruction_color, linestyle="--", linewidth=5, zorder=2)
ax_dec.fill_between(common_grid, decoding_instruction_lower, decoding_instruction_upper, color=instruction_color, alpha=0.2)
ax_dec.plot(common_grid, decoding_base_mean, color=decoding_base_color, linewidth=5, zorder=2, label="   Base Model\n@ critical word")
ax_dec.fill_between(common_grid, decoding_base_lower, decoding_base_upper, color=decoding_base_color, alpha=0.2)
ax_dec.legend(fontsize=14, loc="lower right", bbox_to_anchor=(1, 0.2), framealpha=0.9)
ax_dec.axhline(
    average_significance_threshold,
    linestyle="--",
    color="gray",
    linewidth=3,
    zorder=1,
)
ax_dec.text(
    0.2,
    average_significance_threshold - 0.035,
    f"average significance threshold",
    color="gray",
    fontsize=12,
    # ha="left",
    # va="bottom",
)

ax_align.plot(common_grid, alignment_instruction_mean, color=alignment_color, linewidth=4, zorder=2,
              label="Instruct Model\n@ critical word")
ax_align.fill_between(common_grid, alignment_instruction_lower, alignment_instruction_upper, color=alignment_color, alpha=0.2)
ax_align.plot(common_grid, alignment_mean, color=instruction_color, linestyle="--", linewidth=4, zorder=2,
              label="Instruct Model\n@rating readout")
ax_align.fill_between(common_grid, alignment_lower, alignment_upper, color=instruction_color, alpha=0.2)
ax_align.legend(fontsize=14, loc="lower right", framealpha=0.9)

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
fig.savefig(PLOTS_DIR / "trajectory.png", bbox_inches="tight", dpi=150)
plt.show()

caption = (
    "Average decoding accuracy and peak-normalized alignment trajectories across relative "
    "network depth. Each trajectory is interpolated onto 101 evenly spaced positions "
    "(0 = first hidden-state layer, 1 = last); shaded bands show the 95% CI of the "
    "across-model mean. Left: instruction-tuned models without instructions "
    f"(green, n={len(decoding_curves)}), instruction-tuned models with instructions "
    f"(pink dashed, n={len(decoding_instruction_curves)}), and base/pretrained models "
    f"(orange, n={len(decoding_base_curves)}). The dashed gray line marks the average "
    "permutation-based 95th-percentile significance threshold from the instruction-tuned "
    "models without instructions. Right: instruction-tuned models with instructions "
    f"(green, n={len(alignment_instruction_curves)}) and without instructions "
    f"(pink dashed, n={len(alignment_curves)}). Each model's alignment coefficients "
    "are divided by their own peak within each condition before interpolation and averaging. "
    "Base models contribute only to decoding. Missing instruction-conditioned results "
    "are omitted from their respective averages."
)
with open(PLOTS_DIR / "trajectory_caption.txt", "w", encoding="utf-8") as file:
    file.write(caption + "\n")

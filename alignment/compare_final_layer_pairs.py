"""
Side-by-side comparison of the set-centered final_layer_pairs scatter + OLS fit for two
models, in a single 10x5 figure. A plain-text caption is written alongside the plot.

Reuses compare_max_layer_pairs.py's compare_pairs(), pointed at each model's
models/{model_id}/alignment/final_layer_pairs.csv (produced by make_final_layer_pairs.py,
i.e. the model's FINAL layer rather than its highest-coefficient layer) and saves:
  alignment/plots/{model_id_a}_vs_{model_id_b}_final_layer_pairs.png
  alignment/plots/{model_id_a}_vs_{model_id_b}_final_layer_pairs_caption.txt

Hand-edit MODEL_IDS below before running.
"""

# from compare_max_layer_pairs import compare_pairs

MODEL_IDS = ["internvl3_78b_instruct", "internvl3_1b_instruct"] 

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualize_final_layer_pairs import (
    MODELS_DIR,
    models_lookup,
    CONDITION_COLORS,
    _add_zscored_probability,
    _add_centered_columns,
)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/alignment/plots")
PLOTS_DIR.mkdir(exist_ok=True)


def compare_pairs(
    model_ids,
    pairs_filename,
    layer_description,
    output_suffix,
    condition_legend_index=0,
    file_format="png",
    title_overrides=None,
    fit_legend_label=r"OLS fit ($\beta$={slope:.3f})",
):
    """
    pairs_filename: csv under each model's alignment/ folder to read (e.g.
        "max_layer_pairs.csv" or "final_layer_pairs.csv").
    layer_description: phrase describing which layer that csv was built from, used in the
        caption (e.g. "most human-aligned layer" or "final layer").
    output_suffix: filename suffix for the saved plot/caption (e.g. "max_layer_pairs" or
        "final_layer_pairs").
    condition_legend_index: which panel (0-indexed) gets the congruent/incongruent color
        legend, top left; every panel always gets its own OLS-fit-only legend, bottom right.
    file_format: image format to save the plot as ("png" or "pdf").
    fit_legend_label: format string (with a {slope} field) for each panel's fit-line legend
        label.
    title_overrides: optional {model_id: display_title} to use instead of
        models_lookup.json's title, for this plot only.
    """
    title_overrides = title_overrides or {}
    fig, axes = plt.subplots(1, len(model_ids), figsize=(10, 5), sharey=True)

    slopes = {}
    titles = []

    for ax, model_id in zip(axes, model_ids):
        model_dir = MODELS_DIR / model_id
        title = title_overrides.get(model_id, models_lookup[model_id]["title"])
        titles.append(title)

        pairs_df = pd.read_csv(model_dir / "alignment" / pairs_filename)
        pairs_df = _add_zscored_probability(pairs_df)
        pairs_df = _add_centered_columns(pairs_df)

        condition_handles = []
        for condition, color in CONDITION_COLORS.items():
            condition_df = pairs_df[pairs_df["condition"] == condition]
            handle = ax.scatter(
                condition_df["probability_congruent_z"],
                condition_df["rating"],
                color=color,
                label=condition,
                alpha=0.6,
                edgecolors="none",
                s=110,
            )
            condition_handles.append(handle)

        # slope, intercept = np.polyfit(
        #     pairs_df["probability_congruent_z_centered"], pairs_df["rating_centered"], 1
        # )
        # slopes[title] = slope
        # x_fit = np.array(
        #     [
        #         pairs_df["probability_congruent_z_centered"].min(),
        #         pairs_df["probability_congruent_z_centered"].max(),
        #     ]
        # )
        # y_fit = slope * x_fit + intercept
        # fit_handle, = ax.plot(
        #     x_fit,
        #     y_fit,
        #     color="black",
        #     linewidth=2.5,
        #     linestyle="--",
        #     label=fit_legend_label.format(slope=slope),
        # )

        ax.set_xlabel(r"$P_z$(congruent)", fontsize=19)
        ax.set_title(title, fontsize=22)
        ax.tick_params(axis="both", labelsize=16, width=2, length=8)
        for spine in ax.spines.values():
            spine.set_linewidth(2)
        ax.grid(True, linewidth=1.5, alpha=0.3)

        # Every panel gets an OLS-fit-only legend, bottom right.
        # fit_legend = ax.legend(handles=[fit_handle], loc="lower right", fontsize=19, framealpha=0.9)

        # Only the designated panel also gets the congruent/incongruent color legend, top left.
        if ax is axes[condition_legend_index]:
            # ax.add_artist(fit_legend)
            ax.legend(
                handles=condition_handles,
                labels=list(CONDITION_COLORS.keys()),
                loc="upper left",
                fontsize=15,
                framealpha=0.9,
            )

    axes[0].set_ylabel("Behavioral rating", fontsize=19)

    fig.tight_layout()

    output_stem = "_vs_".join(model_ids) + f"_{output_suffix}"
    output_path = PLOTS_DIR / f"{output_stem}.{file_format}"
    fig.savefig(output_path, dpi=200)
    print(f"Comparison plot for {', '.join(model_ids)} saved to {output_path}.")
    plt.show()

    # caption_path = PLOTS_DIR / f"{output_stem}_caption.txt"
    # caption_path.write_text(_make_caption(titles, slopes, layer_description) + "\n")
    # print(f"Caption saved to {caption_path}.")


if __name__ == "__main__":
    compare_pairs(
        MODEL_IDS,
        "final_layer_pairs.csv",
        "final layer",
        "final_layer_pairs",
        condition_legend_index=1,
        file_format="png",
    )

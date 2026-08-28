"""
For a given model, read models/{model_id}/alignment/final_layer_pairs.csv (produced by
make_final_layer_pairs.py) and scatter-plot decoder probability_congruent (x) against
Behavioral rating (y), colored by ground-truth condition:
  - green: congruent
  - red: incongruent

x/y are oriented this way (probability as the independent variable, rating as the
dependent variable) to match alignment_rep_bev.py's "rating ~ probability_z +
C(item_index)" regression, so that the OLS beta in final_layer_pairs_centered_fit.png
matches the corresponding layer's coefficient in coefficients.csv.

Four versions are produced, each saved under the model's own alignment folder:
  - models/{model_id}/alignment/final_layer_pairs.png: raw probability_congruent
  - models/{model_id}/alignment/final_layer_pairs_zscored.png: probability_congruent
    z-scored across all items within that layer
  - models/{model_id}/alignment/final_layer_pairs_centered.png: rating and z-scored
    probability_congruent, each centered on their own stimulus set's (i.e. the four rows
    sharing a stimulus_index) mean
  - models/{model_id}/alignment/final_layer_pairs_centered_fit.png: same set-centered data
    as above, with an OLS line of best fit overlaid and its beta (slope) labeled

Hand-edit model_id below before running.
"""

from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

CONDITION_COLORS = {
    "congruent": "green",
    "incongruent": "red",
}

# Fixed figure size and axes rect (in figure-fraction coords) shared by both plots, so
# that overlaying the raw and z-scored PDFs lines up the scatter points/axes box exactly.
# This must NOT depend on tick-label content (e.g. via tight_layout() or bbox_inches=
# "tight" at save time), since "-1.0" vs "1.0" tick labels have different widths and
# would otherwise shift the axes box between the two plots.
FIGSIZE = (10, 5)
AXES_RECT = (0.13, 0.20, 0.83, 0.68)  # left, bottom, width, height

def _scatter_final_layer_pairs(pairs_df, x_col, x_label, y_col, y_label, title):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(AXES_RECT)

    for condition, color in CONDITION_COLORS.items():
        condition_df = pairs_df[pairs_df["condition"] == condition]
        ax.scatter(
            condition_df[x_col],
            condition_df[y_col],
            color=color,
            label=condition,
            alpha=0.6,
            edgecolors="none",
            s=150,
        )

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_title(title, fontsize=24)

    ax.tick_params(axis="both", labelsize=22, width=2, length=8)

    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.legend(fontsize=17.5, loc="best", framealpha=0.9)
    ax.grid(True, linewidth=2, alpha=0.3)

    return fig

def _add_zscored_probability(pairs_df):
    pairs_df = pairs_df.copy()
    pairs_df["probability_congruent_z"] = (
        pairs_df["probability_congruent"] - pairs_df["probability_congruent"].mean()
    ) / pairs_df["probability_congruent"].std()
    return pairs_df

def plot_final_layer_pairs(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]

    pairs_df = pd.read_csv(model_dir / "alignment" / "final_layer_pairs.csv")

    fig = _scatter_final_layer_pairs(
        pairs_df, "probability_congruent", "P(congruent)", "rating", "Behavioral rating", title
    )

    plots_dir = model_dir / "alignment"
    plots_dir.mkdir(exist_ok=True)

    fig.savefig(plots_dir / "final_layer_pairs.png", dpi=200)
    print(f"Final layer pairs plot for {model_id} saved.")
    plt.show()

def plot_final_layer_pairs_zscored(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]

    pairs_df = pd.read_csv(model_dir / "alignment" / "final_layer_pairs.csv")
    pairs_df = _add_zscored_probability(pairs_df)

    fig = _scatter_final_layer_pairs(
        pairs_df,
        "probability_congruent_z",
        r"$P_z$(congruent)",
        "rating",
        "Behavioral rating",
        title,
    )

    plots_dir = model_dir / "alignment"
    plots_dir.mkdir(exist_ok=True)

    fig.savefig(plots_dir / "final_layer_pairs_zscored.png", dpi=200)
    print(f"Z-scored final layer pairs plot for {model_id} saved.")
    plt.show()

def _add_centered_columns(pairs_df):
    pairs_df = pairs_df.copy()
    # Each stimulus set is the four rows (2 image_words x 2 conditions) sharing a
    # stimulus_index. Center that set's four ratings on their own mean, and separately
    # center its four z-scored probabilities on their own mean.
    set_groups = pairs_df.groupby("stimulus_index")
    pairs_df["rating_centered"] = pairs_df["rating"] - set_groups["rating"].transform("mean")
    pairs_df["probability_congruent_z_centered"] = pairs_df[
        "probability_congruent_z"
    ] - set_groups["probability_congruent_z"].transform("mean")
    return pairs_df

def plot_final_layer_pairs_centered(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]

    pairs_df = pd.read_csv(model_dir / "alignment" / "final_layer_pairs.csv")
    pairs_df = _add_zscored_probability(pairs_df)
    pairs_df = _add_centered_columns(pairs_df)

    fig = _scatter_final_layer_pairs(
        pairs_df,
        "probability_congruent_z_centered",
        r"$P_z$(congruent), centered",
        "rating_centered",
        "Behavioral rating, centered",
        title,
    )

    plots_dir = model_dir / "alignment"
    plots_dir.mkdir(exist_ok=True)

    fig.savefig(plots_dir / "final_layer_pairs_centered.png", dpi=200)
    print(f"Set-centered final layer pairs plot for {model_id} saved.")
    plt.show()

def plot_final_layer_pairs_centered_fit(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]

    pairs_df = pd.read_csv(model_dir / "alignment" / "final_layer_pairs.csv")
    pairs_df = _add_zscored_probability(pairs_df)
    pairs_df = _add_centered_columns(pairs_df)

    fig = _scatter_final_layer_pairs(
        pairs_df,
        "probability_congruent_z_centered",
        r"$P_z$(congruent), centered",
        "rating_centered",
        "Behavioral rating, centered",
        title,
    )
    ax = fig.axes[0]

    # rating (y) on probability_z (x), matching alignment_rep_bev.py's regression
    # direction so this beta lines up with coefficients.csv's value for this layer.
    slope, intercept = np.polyfit(
        pairs_df["probability_congruent_z_centered"], pairs_df["rating_centered"], 1
    )
    x_fit = np.array(
        [
            pairs_df["probability_congruent_z_centered"].min(),
            pairs_df["probability_congruent_z_centered"].max(),
        ]
    )
    y_fit = slope * x_fit + intercept
    ax.plot(
        x_fit,
        y_fit,
        color="black",
        linewidth=2.5,
        linestyle="--",
        label=rf"OLS fit ($\beta$={slope:.3f})",
    )

    # Re-draw the legend so it picks up the fit line alongside the condition colors.
    ax.legend(fontsize=17.5, loc="best", framealpha=0.9)

    plots_dir = model_dir / "alignment"
    plots_dir.mkdir(exist_ok=True)

    fig.savefig(plots_dir / "final_layer_pairs_centered_fit.png", dpi=200)
    print(f"Set-centered max layer pairs plot with OLS fit for {model_id} saved.")
    plt.show()


if __name__ == "__main__":
    model_id = "internvl3_78b_instruct"
    plot_final_layer_pairs(model_id)
    plot_final_layer_pairs_zscored(model_id)
    plot_final_layer_pairs_centered(model_id)
    plot_final_layer_pairs_centered_fit(model_id)

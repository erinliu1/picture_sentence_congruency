from pathlib import Path

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/alignment/plots")
PLOTS_DIR.mkdir(exist_ok=True)

def add_ci(df):
    df = df.copy()
    df["lower"] = df["coefficient"] - 1.96 * df["se"]
    df["upper"] = df["coefficient"] + 1.96 * df["se"]
    return df

def plot_alignment(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]
    color = models_lookup[model_id]["color"]
    family = models_lookup[model_id]["family"]

    coefficients_df = pd.read_csv(model_dir / "alignment" / "coefficients.csv")
    coefficients_df = add_ci(coefficients_df)

    plt.figure(figsize=(10, 5))

    plt.plot(
        coefficients_df["layer_index"], 
        coefficients_df["coefficient"], 
        markersize=12, 
        linewidth=4, 
        marker=None, 
        color=color, 
        label=title,
    )

    plt.fill_between(
        coefficients_df["layer_index"],
        coefficients_df["lower"], 
        coefficients_df["upper"], 
        color=color, 
        alpha=0.2
    )

    significant_df = coefficients_df[coefficients_df["significant_fdr"].astype(bool)]
    star_y = 1.68
    plt.scatter(
        significant_df["layer_index"],
        np.full(len(significant_df), star_y),
        marker="*",
        s=180,
        color=color,
        label="FDR-significant layers",
        clip_on=False,
    )

    plt.xlabel("Layer", fontsize=24)
    plt.ylabel(r"$\beta$", fontsize=24)

    max_layer = coefficients_df["layer_index"].max()
    tick_step = 5 if max_layer >= 20 else 2
    plt.xticks(np.arange(0, max_layer + 1, tick_step), fontsize=22)
    plt.ylim(0, 1.76)
    plt.yticks(np.arange(0.25, 1.75, 0.25), fontsize=22)

    ax = plt.gca()
    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(width=2, length=8)

    plt.legend(fontsize=17.5, loc="lower right", framealpha=0.9)
    plt.grid(True, linewidth=2, alpha=0.3)
    plt.tight_layout()

    plots_dir = PLOTS_DIR / family
    plots_dir.mkdir(exist_ok=True)

    plt.savefig(plots_dir / f"{model_id}.pdf", bbox_inches="tight")
    print(f"Alignment plot for {model_id} saved.")
    plt.show()

def get_normalized_layers(layers):
    return (layers - layers.min()) / (layers.max() - layers.min())

def plot_alignment_for_all_models(family):
    models_to_plot = [model_id for model_id, info in models_lookup.items() if info["family"] == family]
    plots_dir = PLOTS_DIR / family
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id in models_to_plot:
        model_dir = MODELS_DIR / model_id
        title = models_lookup[model_id]["title"]
        color = models_lookup[model_id]["color"]
        
        coefficients_df = pd.read_csv(model_dir / "alignment" / "coefficients.csv")
        normalized_layers = get_normalized_layers(coefficients_df["layer_index"])
        ax.plot(
            normalized_layers, 
            coefficients_df["coefficient"], 
            markersize=12, 
            linewidth=4, 
            marker=None, 
            color=color, 
            label=title,
        )

    ax.set_xlabel("Normalized layer", fontsize=24)
    ax.set_ylabel(r"$\beta$", fontsize=24)

    ax.tick_params(
        axis="both",
        labelsize=22,
        width=2,
        length=8,
    )

    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))

    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.arange(0, 2.01, 0.25))

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.grid(
        True,
        linewidth=1.5,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        plots_dir / f"all_{family}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)
    print(f"Alignment plot for all {family} models saved.")

    plot_alignment_legend(family)


def plot_alignment_legend(family):
    models_to_plot = [model_id for model_id, info in models_lookup.items() if info["family"] == family]
    plots_dir = PLOTS_DIR / family
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    handles = []
    for model_id in models_to_plot:
        title = models_lookup[model_id]["title"]
        color = models_lookup[model_id]["color"]
        handle, = ax.plot([], [], color=color, linewidth=4, label=title)
        handles.append(handle)

    legend = ax.legend(
        handles=handles,
        fontsize=17,
        ncol=1,
        framealpha=0.9,
        loc="center",
    )
    ax.axis("off")

    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(
        plots_dir / f"legend_{family}.png",
        dpi=300,
        bbox_inches=bbox.expanded(1.1, 1.1),
    )

    plt.close(fig)
    print(f"Alignment legend for {family} saved.")
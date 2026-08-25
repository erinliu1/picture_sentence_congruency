from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

# Row order, top to bottom.
FAMILY_ORDER = ["qwen", "intern", "plm", "llava", "glm", "gemma3", "gemma4"]


def get_normalized_layers(layers):
    return (layers - layers.min()) / (layers.max() - layers.min())


def models_in_family(family):
    return [model_id for model_id, info in models_lookup.items() if info["family"] == family]


def plot_decoding(ax, family):
    for model_id in models_in_family(family):
        color = models_lookup[model_id]["color"]
        results_dir = MODELS_DIR / model_id / "decoding"
        df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
        ax.plot(
            get_normalized_layers(df["layer_index"]),
            df["all_accuracy"],
            color=color,
            linewidth=4,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.0)
    ax.set_yticks(np.arange(0.5, 1.01, 0.25))


def plot_alignment(ax, family):
    for model_id in models_in_family(family):
        color = models_lookup[model_id]["color"]
        results_dir = MODELS_DIR / model_id / "alignment"
        df = pd.read_csv(results_dir / "coefficients.csv")
        ax.plot(
            get_normalized_layers(df["layer_index"]),
            df["coefficient"],
            color=color,
            linewidth=4,
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2.0)
    ax.set_yticks(np.arange(0, 2.01, 0.5))


def plot_legend(ax, family):
    handles = []
    for model_id in models_in_family(family):
        title = models_lookup[model_id]["title"]
        color = models_lookup[model_id]["color"]
        handle, = ax.plot([], [], color=color, linewidth=5, label=title)
        handles.append(handle)
    ax.legend(
        handles=handles,
        fontsize=17,
        loc="upper left",
        bbox_to_anchor=(0, 1),
        frameon=False,
        handlelength=2,
        labelspacing=0.6,
        borderaxespad=0,
        alignment="left",
    )
    ax.axis("off")


def build_grid():
    n_rows = len(FAMILY_ORDER)
    fig, axes = plt.subplots(
        n_rows, 3,
        figsize=(17, 2.6 * n_rows),
        gridspec_kw={"width_ratios": [3, 3, 1.6]},
        constrained_layout=True,
    )

    for row_i, family in enumerate(FAMILY_ORDER):
        ax_dec, ax_align, ax_leg = axes[row_i]

        plot_decoding(ax_dec, family)
        plot_alignment(ax_align, family)
        plot_legend(ax_leg, family)

        for ax in (ax_dec, ax_align):
            ax.tick_params(axis="both", labelsize=17, width=2.5, length=9)
            ax.grid(True, linewidth=1.8, alpha=0.3)
            for spine in ax.spines.values():
                spine.set_linewidth(2.5)
            if row_i < n_rows - 1:
                ax.set_xticklabels([])

    axes[0, 0].set_title("Decoding accuracy", fontsize=24)
    axes[0, 1].set_title(r"Alignment coefficient ($\beta$)", fontsize=24)
    axes[0, 2].set_title("Models", fontsize=24)

    axes[-1, 0].set_xlabel("Normalized layer", fontsize=20)
    axes[-1, 1].set_xlabel("Normalized layer", fontsize=20)

    fig.savefig(
        MODELS_DIR / "grid_summary.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"Grid summary saved to {MODELS_DIR / 'grid_summary.png'}")


if __name__ == "__main__":
    build_grid()

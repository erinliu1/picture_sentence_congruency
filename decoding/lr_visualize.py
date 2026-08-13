from pathlib import Path

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/plots")
PLOTS_DIR.mkdir(exist_ok=True)

def average_significance_threshold(family):
    models_to_plot = [model_id for model_id, info in models_lookup.items() if info["family"] == family]
    thresholds = []
    for model_id in models_to_plot:
        results_dir = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/decoding")
        significance_df = pd.read_csv(results_dir / "layerwise_significance.csv")
        thresholds.append(significance_df["threshold_95"].iloc[0])
    return np.mean(thresholds)
    
def lr_visualize(model_id):
    title = models_lookup[model_id]["title"]
    color = models_lookup[model_id]["color"]
    family = models_lookup[model_id]["family"]
    
    results_dir = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/decoding")
    main_accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
    bootstrap_accuracy_df = pd.read_csv(results_dir / "all_bootstraps.csv")
    max_stats_df = pd.read_csv(results_dir / "max_stats.csv")
    significance_df = pd.read_csv(results_dir / "layerwise_significance.csv")

    bootstrap_ci = (bootstrap_accuracy_df.groupby("layer_index")["all_accuracy"].quantile([0.025, 0.975]).unstack().reset_index())
    bootstrap_ci.columns = ["layer_index","lower","upper"]
    plot_df = main_accuracy_df.merge(bootstrap_ci, on="layer_index")
    permutation_threshold = significance_df["threshold_95"].iloc[0]

    plt.figure(figsize=(10, 5))
    plt.plot(
        plot_df["layer_index"],
        plot_df["all_accuracy"],
        marker=None,
        color=color,
        markersize=12,
        linewidth=4,
        label=title,
    )

    plt.fill_between(
        plot_df["layer_index"],
        plot_df["lower"],
        plot_df["upper"],
        color=color,
        alpha=0.2,
    )

    plt.axhline(
        permutation_threshold,
        linestyle="--",
        color="gray",
        linewidth=3,
        label=f"significance threshold ({permutation_threshold:.2f})",
    )

    plt.xlabel("Layer", fontsize=24)
    plt.ylabel("Accuracy", fontsize=24)

    max_layer = plot_df["layer_index"].max()
    tick_step = 5 if max_layer >= 20 else 2
    plt.xticks(np.arange(0, max_layer + 1, tick_step), fontsize=22)
    plt.ylim(0.45, 1.0)
    plt.yticks(np.arange(0.5, 1.01, 0.1), fontsize=22)

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

    plt.savefig(f"{plots_dir}/{model_id}.pdf", bbox_inches="tight")
    print(f"Layerwise decoding accuracy plot for model {model_id} saved.")
    plt.close()


def get_normalized_layers(layers):
    return (layers - layers.min()) / (layers.max() - layers.min())

def lr_visualize_all_models(family):
    models_to_plot = [model_id for model_id, info in models_lookup.items() if info["family"] == family]
    plots_dir = PLOTS_DIR / family
    plots_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    for model_id in models_to_plot:
        model_dir = MODELS_DIR / model_id
        title = models_lookup[model_id]["title"]
        color = models_lookup[model_id]["color"]
        results_dir = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/decoding")
        main_accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
        normalized_layers = get_normalized_layers(main_accuracy_df["layer_index"])
        ax.plot(
            normalized_layers, 
            main_accuracy_df["all_accuracy"], 
            markersize=12, 
            linewidth=4, 
            marker=None, 
            color=color, 
            label=title,
        )
    
    ax.axhline(
        average_significance_threshold(family),
        linestyle="--",
        color="black",
        label=f"Average significance threshold ({average_significance_threshold(family):.2f})",
        linewidth=3,
        alpha=0.7,
    )

    ax.set_xlabel("Normalized layer", fontsize=24)
    ax.set_ylabel("Decoding accuracy", fontsize=24)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_ylim(0.45, 1.0)
    ax.tick_params(
        axis="both",
        labelsize=22,
        width=2,
        length=8,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.grid(
        True,
        linewidth=1.5,
        alpha=0.3,
    )

    ax.legend(
        fontsize=17,
        ncol=1,
        framealpha=0.9,
        loc="lower right",
    )
    fig.tight_layout()

    fig.savefig(
        plots_dir / f"all_{family}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)



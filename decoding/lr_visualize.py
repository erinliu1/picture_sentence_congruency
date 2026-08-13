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

def lr_visualize(model_id):
    title = models_lookup[model_id]["title"]
    color = models_lookup[model_id]["color"]
    
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
    plt.savefig(f"{PLOTS_DIR}/{model_id}.pdf", bbox_inches="tight")
    print(f"Layerwise accuracy plot saved to {PLOTS_DIR}/{model_id}.pdf")
    plt.close()
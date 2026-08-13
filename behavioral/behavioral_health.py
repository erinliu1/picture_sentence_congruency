from pathlib import Path

import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")

def extract_health(model_id):
    BEHAVIOR_DIR = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/behavior")
    
    try:
        df = pd.read_csv(BEHAVIOR_DIR / "ratings.csv")
    except:
        print(f"ratings.csv not found for {models_lookup[model_id]['title']}, skipping")
        return
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    for condition in ["congruent", "incongruent"]:
        condition_df = df.query("condition == @condition")
        alpha = 0.7 if condition == "congruent" else 0.6
        weights = np.ones(len(condition_df)) / len(condition_df)

        axs[0].hist(
            condition_df["top_one_token"],
            bins=np.arange(0.5, 6.5, 1),
            weights=weights,
            alpha=alpha,
            label=condition,
            linewidth=2,
            edgecolor="black",
        )

        axs[1].hist(
            condition_df["rating"],
            bins=15,
            weights=weights,
            alpha=alpha,
            label=condition,
            linewidth=2,
            edgecolor="black",
        )

    for ax, subplot_title in zip(axs, ["Top-One Token Distribution", "Expected Rating Distribution"]):
        ax.spines["top"].set_linewidth(2)
        ax.spines["right"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

        ax.tick_params(width=2, length=8)
        ax.tick_params(axis="x", labelsize=22)
        ax.tick_params(axis="y", labelsize=22)

        ax.grid(True, linewidth=2, alpha=0.3)

        ax.set_title(subplot_title, fontsize=22)
        ax.set_xlabel("Rating", fontsize=22)
        ax.set_ylabel("Proportion", fontsize=22)
        ax.legend(loc="best", frameon=True, fontsize=22)

    axs[0].set_xticks(range(1, 6))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"health_{model_id}.pdf", bbox_inches="tight")
    plt.close(fig)

    summary_rows = []
    for condition in ["congruent", "incongruent"]:
        condition_df = df.query("condition == @condition")
        row = {
            "condition": condition,
            "mean_rating": condition_df["rating"].mean(),
            "std_rating": condition_df["rating"].std(),
            "mean_top_one_token": condition_df["top_one_token"].mean(),
            "std_top_one_token": condition_df["top_one_token"].std(),
        }
        for value in range(1, 6):
            row[f"proportion_top_one_token_{value}"] = (condition_df["top_one_token"] == value).mean()
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(BEHAVIOR_DIR / "health.csv", index=False) 
    print(f"Health results for {models_lookup[model_id]['title']} saved to", BEHAVIOR_DIR / "health.csv")


def save_all_health():
    for model_id in sorted(models_lookup.keys()):
        extract_health(model_id)
    print("Health results saved to", PLOTS_DIR)


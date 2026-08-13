from pathlib import Path

import json
import pandas as pd
import matplotlib.pyplot as plt

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/confidence/plots")
PLOTS_DIR.mkdir(exist_ok=True)
    
def plot_confidence(vlm_df, title, plot_dir):
    vlm_df["prediction"] = (vlm_df["rating"] > 3).map({True: "congruent", False: "incongruent"})
    vlm_df["correct"] = (vlm_df["prediction"] == vlm_df["condition"])

    plt.figure(figsize=(10, 5))

    plt.hist(
        vlm_df.query("condition=='congruent'")["max_probability"],
        bins=15,
        alpha=0.7,
        label="congruent",
        linewidth=2,
        edgecolor="black",

    )

    plt.hist(
        vlm_df.query("condition=='incongruent'")["max_probability"],
        bins=15,
        alpha=0.6,
        label="incongruent",
        linewidth=2,
        edgecolor="black",

    )
    ax = plt.gca()

    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.tick_params(width=2, length=8)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(True, linewidth=2, alpha=0.3)

    plt.title(f"{title}", fontsize=22)
    plt.xlabel("Maximum probability", fontsize=22)
    plt.ylabel("Count", fontsize=22)
    plt.legend(frameon=True, fontsize=22)
    plt.tight_layout()

    plt.savefig(plot_dir / f"confidence_{model_id}.pdf", bbox_inches="tight")

for model_id in sorted(models_lookup.keys()):
    title = models_lookup[model_id]["title"]
    family = models_lookup[model_id]["family"]
    model_dir = MODELS_DIR / model_id
    try:
        vlm_csv_path = model_dir / "behavior" / "ratings.csv"
        vlm_df = pd.read_csv(vlm_csv_path)
    except:
        continue
    plot_dir = PLOTS_DIR / family
    plot_dir.mkdir(exist_ok=True)   
    plot_confidence(vlm_df, title, plot_dir)
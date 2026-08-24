from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def plot_confusion(ax, df, title):
    y_true = (df["condition"] == "congruent")
    y_pred = df["rating"] > 3

    cm = confusion_matrix(y_true, y_pred, labels=[True, False])

    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rating > 3", "Rating ≤ 3"])

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actually\nCongruent", "Actually\nIncongruent"])

    ax.tick_params(axis="both", labelsize=18, width=2, length=8)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    ax.set_title(title, fontsize=24)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=24, color="white" if cm[i, j] > cm.max() / 2 else "black")
    return im

def compute_accuracy(df, condition=None):
    # accuracy is computed as the proportion of correct row-level predictions (320 rows, one per sentence-picture pair)
    # condition="congruent"/"incongruent" restricts to that subset only; condition=None uses all rows.
    if condition is not None:
        df = df.query("condition == @condition")
    y_true = df["condition"] == "congruent"
    y_pred = df["rating"] > 3
    return (y_true == y_pred).mean()

def extract_accuracy_VLM(model_id):
    title = models_lookup[model_id]["title"]
    family = models_lookup[model_id]["family"]
    model_dir = MODELS_DIR / model_id
    try:
        vlm_df = pd.read_csv(model_dir / "behavior" / "ratings.csv")
    except:
        return
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    plot_dir = PLOTS_DIR / family
    plot_dir.mkdir(exist_ok=True)

    plot_confusion(ax, vlm_df, title)
    fig.savefig(plot_dir / f"cm_{model_id}.pdf", bbox_inches="tight")
    print(f"Confusion matrix for {model_id} saved.")
    plt.close()
    return {
        "accuracy": compute_accuracy(vlm_df),
        "congruent_accuracy": compute_accuracy(vlm_df, "congruent"),
        "incongruent_accuracy": compute_accuracy(vlm_df, "incongruent"),
    }

def save_all_accuracy():
    accuracy_records = []
    for model_id in models_lookup.keys():
        result = extract_accuracy_VLM(model_id)
        if result is not None:
            accuracy_records.append({"source": model_id, **result})
    accuracy_df = pd.DataFrame(accuracy_records)
    accuracy_df.to_csv(RESULTS_DIR / "accuracy.csv", index=False)
    print("All accuracy results saved.")
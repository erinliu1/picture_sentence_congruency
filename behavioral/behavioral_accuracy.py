from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

human_df = pd.read_csv(HUMAN_PATH)
human_df = (human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean())

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

def compute_accuracy(df):
    # accuracy is computed as the proportion of correct row-level predictions (320 rows, one per sentence-picture pair)
    y_true = df["condition"] == "congruent"
    y_pred = df["rating"] > 3
    return (y_true == y_pred).mean()

def extract_accuracy_VLM(model_id):
    title = models_lookup[model_id]["title"]
    model_dir = MODELS_DIR / model_id
    try:
        vlm_df = pd.read_csv(model_dir / "behavior" / "ratings.csv")
    except:
        return
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    plot_confusion(ax, vlm_df, title)
    fig.savefig(PLOTS_DIR / f"cm_{model_id}.pdf", bbox_inches="tight")
    print(f"Confusion matrix for {title} saved to", PLOTS_DIR / f"cm_{model_id}.pdf")
    plt.close()
    return compute_accuracy(vlm_df)

def extract_accuracy_human():
    title = 'Human'
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    plot_confusion(ax, human_df, title)
    fig.savefig(PLOTS_DIR / f"cm_human.pdf", bbox_inches="tight")
    print("Confusion matrix for human ratings saved to", PLOTS_DIR / f"cm_human.pdf")
    plt.close()
    return compute_accuracy(human_df)

def save_all_accuracy():
    accuracy_records = []
    for model_id in sorted(models_lookup.keys()):
        accuracy_records.append({"source": model_id, "accuracy": extract_accuracy_VLM(model_id)})
    accuracy_records.append({"source": "human", "accuracy": extract_accuracy_human()})
    accuracy_df = pd.DataFrame(accuracy_records)
    accuracy_df.to_csv(RESULTS_DIR / "accuracy.csv", index=False)
    print("Accuracy results saved to", RESULTS_DIR / "accuracy.csv")


from pathlib import Path
from behavioral_accuracy import plot_confusion, compute_accuracy
import pandas as pd
from scipy.stats import spearmanr

HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")
human_df = pd.read_csv(HUMAN_PATH)
human_df = (human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean())

def plot_confusion_human():
    title = 'Human'
    fig, ax = plt.subplots(figsize=(5, 6), constrained_layout=True)
    plot_confusion(ax, human_df, title)
    fig.savefig(PLOTS_DIR / f"cm_human.png", bbox_inches="tight", dpi=600)
    plt.close()

def get_human_accuracy():
    return {
        "accuracy": compute_accuracy(human_df),
        "congruent_accuracy": compute_accuracy(human_df, "congruent"),
        "incongruent_accuracy": compute_accuracy(human_df, "incongruent"),
    }

def count_congruent_dominant_items():
    def item_separates(group):
        congruent_ratings = group.loc[group["condition"] == "congruent", "rating"]
        incongruent_ratings = group.loc[group["condition"] == "incongruent", "rating"]
        return congruent_ratings.min() > incongruent_ratings.max()

    return human_df.groupby("item_index").apply(item_separates).sum()


# print(count_congruent_dominant_items())

# calculate the spearman's correlation (as in behavioral_correlation.py) between the human ratings and as if you assigned 0's to every single incongruent sample and 1's to every single congruent sample.
def get_human_condition_correlation():
    condition_labels = (human_df["condition"] == "congruent").astype(int)
    rho, p = spearmanr(human_df["rating"], condition_labels)
    return rho, p

print(get_human_condition_correlation())
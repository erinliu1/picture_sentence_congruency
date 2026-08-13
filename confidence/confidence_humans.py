from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")
human_df = pd.read_csv(HUMAN_PATH)
human_df = (human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean())

PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/confidence/plots")
PLOTS_DIR.mkdir(exist_ok=True)

### confidence analysis
human_df["confidence"] = human_df["rating"] - 3
human_df.loc[human_df["condition"] == "incongruent", "confidence"] *= -1

confidence_counts = (
    human_df.groupby("condition")["confidence"]
    .value_counts()
    .unstack(fill_value=0)
    .reindex(columns=[-2, -1, 0, 1, 2])
)

# Convert counts to proportions within each condition
confidence_proportions = confidence_counts.div(
    confidence_counts.sum(axis=1),
    axis=0,
).T

ax = confidence_proportions.plot(
    kind="bar",
    figsize=(10, 5),
    width=0.75,
    edgecolor="black",
    linewidth=2,
)
ax.tick_params(axis="y", labelsize=22)

ax.set_xlabel("Confidence", fontsize=22)
ax.set_ylabel("Proportion of responses", fontsize=22)
ax.set_title("Humans", fontsize=22)
ax.set_xticklabels(
    ["−2", "−1", "0", "1", "2"],
    rotation=0, fontsize=22,
)

ax.legend(
    frameon=True,
    fontsize=22,
)

ax = plt.gca()

ax.spines["top"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)

ax.tick_params(width=2, length=8)
plt.grid(True, linewidth=2, alpha=0.3)


plt.tight_layout()
plt.savefig(PLOTS_DIR / "confidence_humans.pdf", bbox_inches="tight")
plt.show()
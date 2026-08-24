from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HUMAN_PATH = Path(
    "/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv"
)
human_df = pd.read_csv(HUMAN_PATH)

PLOTS_DIR = Path(
    "/Intern/Erin/picture_sentence_congruency/confidence/plots"
)
PLOTS_DIR.mkdir(exist_ok=True)


### confidence analysis

# Compute confidence for each individual participant response
human_df["confidence"] = human_df["rating"] - 3

# Reverse the sign for incongruent trials so that:
# +2 = maximal confidence in the assigned condition
# -2 = maximal confidence against the assigned condition
human_df.loc[
    human_df["condition"] == "incongruent",
    "confidence"
] *= -1

confidence_counts = (
    human_df.groupby("condition")["confidence"]
    .value_counts()
    .unstack(fill_value=0)
    .reindex(columns=[-2, -1, 0, 1, 2], fill_value=0)
)

# Convert counts to proportions within each condition
confidence_proportions = confidence_counts.div(
    confidence_counts.sum(axis=1),
    axis=0,
).T * 100

ax = confidence_proportions.plot(
    kind="bar",
    figsize=(7, 4),
    width=0.75,
    edgecolor="black",
    linewidth=2,
)

ax.set_xlabel("Confidence", fontsize=22)
ax.set_ylabel("% of responses", fontsize=22)

ax.set_xticklabels(
    ["−2", "−1", "0", "1", "2"],
    rotation=0,
    fontsize=22,
)

ax.tick_params(axis="y", labelsize=22)

ax.legend(
    frameon=True,
    fontsize=22,
)

for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.tick_params(width=2, length=8)
ax.grid(True, linewidth=2, alpha=0.3)

plt.tight_layout()
plt.savefig(
    PLOTS_DIR / "confidence_humans.png",
    bbox_inches="tight",
    dpi=600,
)
plt.show()


# count how many responses were made in the positive confidence range (1 or 2) out of the total number of responses, separately for each condition
positive_counts = human_df.groupby("condition")["confidence"].apply(
    lambda s: s.isin([2]).sum()
)
total_counts = human_df.groupby("condition")["confidence"].size()
positive_proportions = positive_counts / total_counts * 100

for condition in positive_counts.index:
    print(
        f"{condition}: {positive_counts[condition]} / {total_counts[condition]} "
        f"({positive_proportions[condition]:.2f}%)"
    )

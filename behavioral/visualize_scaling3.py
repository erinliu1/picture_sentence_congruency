import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from pathlib import Path
import pandas as pd
import numpy as np

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

BEHAVIORAL_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
accuracy_df = pd.read_csv(BEHAVIORAL_DIR / "accuracy.csv")
concordance_df = pd.read_csv(BEHAVIORAL_DIR / "concordance_scores.csv")
correlation_df = pd.read_csv(BEHAVIORAL_DIR / "correlation.csv")

family_lookup = {
    "qwen": {"label": "Qwen3-VL"},
    "intern": {"label": "InternVL3"},
    "plm": {"label": "Perception-LM"},
    "llava": {"label": "LLaVA-OneVision"},
    "glm": {"label": "GLM-4.6V"},
    "gemma3": {"label": "Gemma3"},
    "gemma4": {"label": "Gemma4"},
}

family_colors = {
    "qwen": "#67ae00",     # green
    "intern": "#2563EB",   # blue
    "plm": "#8B5CF6",      # violet
    "llava": "#4fd6ff",    # pink/magenta
    "glm": "#EC4899",      # cyan
    "gemma3": "#ffd42b",   # amber
    "gemma4": "#DC2626",   # red-orange
}
family_linestyles = {
    "glm": "--",
    "gemma4": ":",
    "intern": ":",
    "gemma3": "--"
}

plot_df = accuracy_df.merge(concordance_df, on="source").merge(correlation_df, on="source")
plot_df["family"] = plot_df["source"].map(lambda x: models_lookup[x]["family"])
plot_df["family"] = pd.Categorical(plot_df["family"], categories=family_lookup.keys(), ordered=True)
plot_df["size"] = plot_df["source"].map(lambda x: models_lookup[x]["size"])
# Leading number in the size string, in billions of parameters (e.g. "30B-A3B"
# -> 30, "106-A12B" -> 106, "E2B" -> 2). Used as the log x-axis position.
plot_df["params_b"] = plot_df["size"].str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
plot_df = plot_df.sort_values(by=["family", "params_b"]).reset_index(drop=True)

LABEL_FS = 21
TICK_FS = 20

fig, (ax_acc, ax_conc, ax_rho) = plt.subplots(
    1, 3, figsize=(10, 4.6), sharex=True, gridspec_kw={"wspace": 0.28}
)

for family, family_df in plot_df.groupby("family", observed=True, sort=False):
    family_df = family_df.sort_values("params_b")
    color = family_colors[family]
    linestyle = family_linestyles.get(family, "-")

    ax_acc.plot(family_df["params_b"], family_df["accuracy"], color=color, linestyle=linestyle, linewidth=3.5, zorder=2)
    ax_conc.plot(family_df["params_b"], family_df["concordance"], color=color, linestyle=linestyle, linewidth=3.5, zorder=2)
    ax_rho.plot(family_df["params_b"], family_df["spearman_rho"], color=color, linestyle=linestyle, linewidth=3.5, zorder=2)

for ax in (ax_acc, ax_conc, ax_rho):
    ax.set_xscale("log")
    ax.tick_params(axis="both", labelsize=TICK_FS, width=1.5, length=6)
    ax.tick_params(axis="x", which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.grid(True, linewidth=1, alpha=0.3)

ax_acc.set_title("Accuracy", fontsize=LABEL_FS, pad=10)
ax_conc.set_title("Concordance", fontsize=LABEL_FS, pad=10)
ax_rho.set_title("Spearman's ρ", fontsize=LABEL_FS, pad=10)

family_handles = [
    Line2D(
        [0], [0],
        color=family_colors[family],
        linestyle=family_linestyles.get(family, "-"),
        linewidth=3,
        label=info["label"],
    )
    for family, info in family_lookup.items()
]

fig.legend(
    handles=family_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=4,
    fontsize=18,
    framealpha=0.9,
)

fig.subplots_adjust(left=0.04, right=0.99, top=0.9, bottom=0.32)

plt.savefig(PLOTS_DIR / "scaling3.pdf", bbox_inches="tight")
plt.show()

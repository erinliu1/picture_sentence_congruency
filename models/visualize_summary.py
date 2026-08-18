import matplotlib.pyplot as plt
import json
from pathlib import Path
import pandas as pd
import numpy as np

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

BEHAVIORAL_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
accuracy_df = pd.read_csv(BEHAVIORAL_DIR / "accuracy.csv")
correlation_df = pd.read_csv(BEHAVIORAL_DIR / "correlation.csv")

DECODING_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/results")
decoding_df = pd.read_csv(DECODING_DIR / "summary.csv")

family_lookup = {
    "qwen": {
        "color": "#F0FDCB",   # light lime green
        "label": "Qwen3-VL",
    },
    "intern": {
        "color": "#E8F2FF",   # light blue
        "label": "InternVL3",
    },
    "plm": {
        "color": "#F3EEFF",   # light purple
        "label": "Perception-LM",
    },
    "llava": {
        "color": "#FDEBF4",   # light pink
        "label": "LLaVA-OneVision",
    },
    "glm": {
        "color": "#E5FAFD",   # light cyan
        "label": "GLM-4.6V",
    },
    "gemma3": {
        "color": "#FFF5D6",   # light yellow/gold
        "label": "Gemma3",
    },
    "gemma4": {
        "color": "#FFF0E3",   # light orange
        "label": "Gemma4",
    },
}
type_lookup = {"PLE": "Dense\n(PLE)", "dense": "Dense", "moe": "MoE"}

plot_df = accuracy_df.merge(correlation_df, on="source").merge(decoding_df, on="source")
plot_df["title"] = plot_df["source"].map(lambda x: models_lookup[x]["title"])
plot_df["family"] = plot_df["source"].map(lambda x: models_lookup[x]["family"])
plot_df["family"] = pd.Categorical(plot_df["family"], categories=family_lookup.keys(), ordered=True)
plot_df["type"] = plot_df["source"].map(lambda x: models_lookup[x]["type"])
plot_df["type"] = pd.Categorical(plot_df["type"], categories=type_lookup.keys(), ordered=True)
plot_df["size"] = plot_df["source"].map(lambda x: models_lookup[x]["size"])
plot_df["size_sort"] = plot_df["size"].str.extract(r"^(\d+(?:\.\d+)?)")[0].astype(float)
plot_df = plot_df.sort_values(by=["family", "type", "size_sort"]).reset_index(drop=True)
plot_df["x_label"] = plot_df.apply(lambda row: f'{row["size"]}\n{type_lookup[row["type"]]}', axis=1)

behavior_color = "#222222"          # black
peak_decoding_color = "#2A9D8F"     # dark teal
last_layer_decoding_color = "tab:blue"     
correlation_color = "#B05A7A"       # muted rose

n_families = len(plot_df["family"].unique())
fig, ax = plt.subplots(figsize=(6*n_families, 5))
x = np.arange(len(plot_df))

labels = False
offset = 0.07

for family_i, (family, family_df) in enumerate(
    plot_df.groupby("family", observed=True, sort=False)
):
    family_indices = family_df.index.to_numpy()

    family_left = family_indices.min() - 0.5
    family_right = family_indices.max() + 0.5
    family_center = (family_left + family_right) / 2

    # Family background shading
    ax.axvspan(
        family_left,
        family_right,
        color=family_lookup[family]["color"],
        alpha=0.5,
        zorder=0,
    )

    # Family label
    ax.text(
        family_center,
        1.1,
        family_lookup[family]["label"],
        ha="center",
        va="top",
        fontsize=22,
        transform=ax.get_xaxis_transform(),
        zorder=3,
    )

    # Plot each type separately so lines do not cross type boundaries
    for model_type, type_df in family_df.groupby(
        "type",
        observed=True,
        sort=False,
    ):
        indices = type_df.index.to_numpy()

        # Separator at start of each type block
        ax.axvline(
            indices.min() - 0.5,
            color="grey",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )

        # -------------------------
        # CENTERED CONNECTING LINES
        # -------------------------

        ax.plot(
            indices,
            type_df["accuracy"],
            color=behavior_color,
            linewidth=4,
            label="Behavioral accuracy" if not labels else None,
            zorder=3,
        )

        ax.plot(
            indices,
            type_df["peak_decoding_accuracy"],
            color=peak_decoding_color,
            linewidth=4,
            linestyle="--",
            label="Peak decoding accuracy" if not labels else None,
            zorder=3,
        )

        ax.plot(
            indices,
            type_df["last_layer_decoding_accuracy"],
            color=last_layer_decoding_color,
            linewidth=4,
            linestyle="--",
            label="Last layer decoding accuracy" if not labels else None,
            zorder=3,
        )

        ax.plot(
            indices,
            type_df["spearman_rho"],
            color=correlation_color,
            linewidth=4,
            linestyle=":",
            label="Human–VLM behavioral correlation" if not labels else None,
            zorder=3,
        )

        # -------------------------
        # OFFSET MARKERS ONLY
        # -------------------------

        if len(indices) == 1:
            
            ax.scatter(
                indices - 1.5 * offset,
                type_df["accuracy"],
                color=behavior_color,
                s=100,
                zorder=5,
            )

            ax.scatter(
                indices - 0.5 * offset,
                type_df["peak_decoding_accuracy"],
                color=peak_decoding_color,
                s=100,
                zorder=5,
            )

            ax.scatter(
                indices + 0.5 * offset,
                type_df["last_layer_decoding_accuracy"],
                color=last_layer_decoding_color,
                s=100,
                zorder=5,
            )

            ax.scatter(
                indices + 1.5 * offset,
                type_df["spearman_rho"],
                color=correlation_color,
                s=100,
                zorder=5,
            )

        # Correlation CI remains centered on true model position
        ax.fill_between(
            indices,
            type_df["ci_lower"],
            type_df["ci_upper"],
            color=correlation_color,
            alpha=0.2,
            zorder=1,
        )

        labels = True

# Final separator after last model
ax.axvline(
    len(plot_df) - 0.5,
    color="grey",
    linestyle="--",
    linewidth=2,
    alpha=0.8,
    zorder=2,
)

ax.set_xticks(x)
ax.set_xticklabels(plot_df["x_label"], fontsize=22)

ax.set_ylabel("Performance", fontsize=22)
ax.set_ylim(0.1, 1.05)
ax.set_yticks(np.arange(0.2, 1.05, 0.2))
ax.tick_params(axis="both", labelsize=22, width=2, length=8)
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.legend(fontsize=17.5, loc="best", framealpha=0.9)
plt.grid(True, linewidth=2, alpha=0.3)
plt.tight_layout()

plt.savefig(MODELS_DIR / "summary.pdf", bbox_inches="tight",)
plt.show()
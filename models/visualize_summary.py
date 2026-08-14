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
        "color": "#e0ffc2",
        "label": "Qwen3-VL",
    },
    "intern": {
        "color": "#d8e9ff",
        "label": "InternVL3",
    },
}
type_lookup = {"dense": "Dense", "moe": "MoE"}

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
fig, ax = plt.subplots(figsize=(10*n_families, 5))
x = np.arange(len(plot_df))

labels = False
for family_i, (family, family_df) in enumerate(plot_df.groupby("family", observed=True, sort=False)):
    indices = family_df.index.to_numpy()
    left = indices.min() - 0.5
    right = indices.max() + 0.5
    center = (left + right) / 2

    ax.axvline(
        left,
        color="grey",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
        zorder=2,
    )

    ax.axvspan(
        indices.min() - 0.5,
        indices.max() + 0.5,
        color=family_lookup[family]["color"],
        alpha=0.5,
        zorder=0,
    )

    ax.text(
        center,
        0.6,
        family_lookup[family]["label"],
        ha="center",
        va="top",
        fontsize=22,
        transform=ax.get_xaxis_transform(),
        zorder=3,
    )

    ax.plot(
        indices,
        family_df["accuracy"],
        marker=None,
        color=behavior_color,
        markersize=12,
        linewidth=4,
        label="Behavioral accuracy" if not labels else None,
    )

    ax.plot(
        indices,
        family_df["peak_decoding_accuracy"],
        marker=None,
        color=peak_decoding_color,
        markersize=12,
        linewidth=4,
        linestyle="--",
        label="Peak decoding accuracy" if not labels else None,
    )

    ax.plot(
        indices,
        family_df["last_layer_decoding_accuracy"],
        marker=None,
        color=last_layer_decoding_color,
        markersize=12,
        linewidth=4,
        linestyle="--",
        label="Last layer decoding accuracy" if not labels else None,
    )

    ax.plot(
        indices,
        family_df["spearman_rho"],
        marker=None,
        markersize=12,
        linewidth=4,
        color=correlation_color,
        linestyle=":",
        label="Human–VLM behavioral correlation" if not labels else None,
    )

    ax.fill_between(
        indices,
        family_df["ci_lower"],
        family_df["ci_upper"],
        color=correlation_color,
        alpha=0.2,
    )
    labels = True

ax.set_xticks(x)
ax.set_xticklabels(plot_df["x_label"], fontsize=22)

ax.set_ylabel("Performance", fontsize=22)
ax.set_ylim(0.45, 1)
ax.tick_params(axis="both", labelsize=22, width=2, length=8)
for spine in ax.spines.values():
    spine.set_linewidth(2)

ax.legend(fontsize=17.5, loc="lower right", framealpha=0.9)
plt.grid(True, linewidth=2, alpha=0.3)
plt.tight_layout()

plt.savefig(MODELS_DIR / "summary.pdf", bbox_inches="tight",)
plt.show()
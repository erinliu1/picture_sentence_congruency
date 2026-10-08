import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

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

plot_df = accuracy_df.merge(concordance_df, on="source").merge(correlation_df, on="source")
plot_df = plot_df[plot_df["source"].isin(models_lookup)].copy()
plot_df["family"] = plot_df["source"].map(lambda x: models_lookup[x]["family"])
plot_df["family"] = pd.Categorical(plot_df["family"], categories=family_lookup.keys(), ordered=True)
plot_df = plot_df[plot_df["family"].notna()].copy()
plot_df["size"] = plot_df["source"].map(lambda x: models_lookup[x]["size"])
# Leading number in the size string, in billions of parameters (e.g. "30B-A3B"
# -> 30, "106-A12B" -> 106, "E2B" -> 2). Used as the log x-axis position.
plot_df["params_b"] = plot_df["size"].str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
plot_df = plot_df.sort_values(by=["family", "params_b"]).reset_index(drop=True)

LABEL_FS = 22
TICK_FS = 20
LEGEND_FS = 22

fig, (ax_acc, ax_conc, ax_rho) = plt.subplots(
    1, 3, figsize=(9, 5.56), sharex=True, gridspec_kw={"wspace": 0.28}
)

for family, family_df in plot_df.groupby("family", observed=True, sort=False):
    family_df = family_df.sort_values("params_b")
    color = family_colors[family]

    for ax, metric in ((ax_acc, "accuracy"), (ax_conc, "concordance"), (ax_rho, "spearman_rho")):
        points = family_df[["params_b", metric]].dropna()
        x = points["params_b"].to_numpy()
        y = points[metric].to_numpy()
        if len(x) >= 3:
            # Match visualize_scaling_accuracy.py: shape-preserving interpolation
            # through measured values, performed in logarithmic size space.
            log_x = np.log10(x)
            n_points = max(int((log_x[-1] - log_x[0]) * 60), 100)
            log_x_smooth = np.linspace(log_x[0], log_x[-1], n_points)
            x_smooth = 10 ** log_x_smooth
            y_smooth = PchipInterpolator(log_x, y)(log_x_smooth)
        else:
            x_smooth, y_smooth = x, y
        ax.plot(x_smooth, y_smooth, color=color, linestyle="-", linewidth=3,
                alpha=0.65, zorder=2)
        ax.scatter(x, y, color=color, s=40, alpha=0.8, edgecolors="none",
                   zorder=3)

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

fig.supxlabel("Model Size (Billions of Parameters)", fontsize=22, y=0.31)

family_handles = [
    Line2D(
        [0], [0],
        color=family_colors[family],
        linestyle="-",
        linewidth=3,
        alpha=0.8,
        marker="o",
        markeredgecolor="none",
        markersize=6,
        label=info["label"],
    )
    for family, info in family_lookup.items()
]

fig.legend(
    handles=family_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.004),
    ncol=3,
    fontsize=LEGEND_FS,
    framealpha=0.9,
)

fig.subplots_adjust(left=0.04, right=0.99, top=0.9159, bottom=0.4673)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / "scaling_smoothed.pdf", bbox_inches="tight")
fig.savefig(PLOTS_DIR / "scaling_smoothed.png", bbox_inches="tight", dpi=600)
caption = """Behavioral performance as a function of model size. Panels show classification accuracy (left), within-item concordance (center), and Spearman rank correlation with mean human ratings (right) for 28 instruction-tuned vision-language models across seven families. Each model is evaluated on 320 picture-sentence pairs from 80 stimulus sets; individual models are shown as small, partially transparent points without borders. Accuracy classifies a pair as congruent when its expected 1-5 rating is greater than 3 and incongruent otherwise. Concordance is the proportion of the four within-item comparisons in which a congruent pair receives a strictly higher rating than an incongruent pair, averaged over items; ties do not count as concordant. Spearman's rho compares model and human ratings across matched picture-sentence pairs. Colors identify model families. All curves are solid and partially transparent. The horizontal axis is logarithmic and uses the leading numeric value of the registered model-size label in billions (total rather than active parameters for MoE models). Curves use shape-preserving PCHIP interpolation in log-size space for families with at least three models; two-model families are joined by straight segments. Curves are visual guides through measured model values, not fitted scaling laws. No uncertainty intervals are shown.
"""
(PLOTS_DIR / "scaling_smoothed.txt").write_text(caption, encoding="utf-8")
plt.close(fig)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
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
    "glm": "solid",
    "gemma4": "solid",
    "intern": "solid",
    "gemma3": "solid"
}

plot_df = accuracy_df.copy()
plot_df = plot_df[plot_df["source"].isin(models_lookup)]
plot_df["family"] = plot_df["source"].map(lambda x: models_lookup[x]["family"])
plot_df["family"] = pd.Categorical(plot_df["family"], categories=family_lookup.keys(), ordered=True)
plot_df = plot_df[plot_df["family"].notna()]
plot_df["size"] = plot_df["source"].map(lambda x: models_lookup[x]["size"])
# Leading number in the size string, in billions of parameters (e.g. "30B-A3B"
# -> 30, "106-A12B" -> 106, "E2B" -> 2). Used as the log x-axis position.
plot_df["params_b"] = plot_df["size"].str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
plot_df = plot_df.sort_values(by=["family", "params_b"]).reset_index(drop=True)

LABEL_FS = 24
TICK_FS = 24
LEGEND_FS = 22

fig, ax_acc = plt.subplots(1, 1, figsize=(9, 8.5))

# Points-per-decade density for the smoothed curve, in log10(params_b) space.
SMOOTH_POINTS_PER_DECADE = 60

for family, family_df in plot_df.groupby("family", observed=True, sort=False):
    family_df = family_df.sort_values("params_b")
    color = family_colors[family]
    linestyle = family_linestyles.get(family, "-")

    x = family_df["params_b"].to_numpy()
    y = family_df["accuracy"].to_numpy()

    if len(x) >= 3:
        # Monotone cubic (PCHIP) interpolation in log-x space gives a smooth
        # curve through every real data point without overshoot/ringing
        # between them (unlike a plain cubic spline).
        log_x = np.log10(x)
        interpolator = PchipInterpolator(log_x, y)
        n_points = max(int((log_x[-1] - log_x[0]) * SMOOTH_POINTS_PER_DECADE), 100)
        log_x_smooth = np.linspace(log_x[0], log_x[-1], n_points)
        x_smooth = 10 ** log_x_smooth
        y_smooth = interpolator(log_x_smooth)
    else:
        # Too few points to interpolate meaningfully; fall back to a
        # straight segment (still "smooth" in the sense of no jaggedness).
        x_smooth, y_smooth = x, y

    ax_acc.plot(x_smooth, y_smooth, color=color, linestyle=linestyle, linewidth=9, zorder=2)
    ax_acc.scatter(x, y, color=color, s=140, zorder=3, edgecolors="white", linewidths=2)

HUMAN_ACCURACY = 0.988
human_line = ax_acc.axhline(HUMAN_ACCURACY, color="grey", linestyle="--", linewidth=4, zorder=1)

ax_acc.set_xscale("log")
ax_acc.set_ylabel("Behavioral Accuracy", fontsize=LABEL_FS)
ax_acc.tick_params(axis="both", labelsize=TICK_FS, width=3, length=10)
ax_acc.tick_params(axis="x", which="minor", length=0)
for spine in ax_acc.spines.values():
    spine.set_linewidth(3)
ax_acc.grid(True, linewidth=1.5, alpha=0.3)

fig.supxlabel("Model Size (Billions of Parameters)", fontsize=22, y=0.21)

fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.32)

plt.savefig(PLOTS_DIR / "scaling_accuracy.png", bbox_inches="tight", dpi=600)
plt.show()


def save_standalone_legend(handles, ncol, out_path, fontsize=18, height=None):
    """Render just a legend (no axes/data) to its own tightly-cropped PNG.
    If `height` is given (inches), the saved PNG is exactly that tall (legend vertically
    centered within it) rather than cropped tight to the legend's own extent."""
    fig_legend = plt.figure(figsize=(0.1, 0.1))
    legend = fig_legend.legend(handles=handles, loc="center", ncol=ncol, fontsize=fontsize, framealpha=0)
    fig_legend.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig_legend.dpi_scale_trans.inverted())
    if height is not None:
        y_center = (bbox.y0 + bbox.y1) / 2
        bbox = Bbox.from_extents(bbox.x0, y_center - height / 2, bbox.x1, y_center + height / 2)
    fig_legend.savefig(out_path, dpi=600, bbox_inches=bbox)
    plt.close(fig_legend)


family_handles = [
    Line2D(
        [0], [0],
        color=family_colors[family],
        linestyle=family_linestyles.get(family, "-"),
        linewidth=9,
        label=info["label"],
    )
    for family, info in family_lookup.items()
]
human_handle = Line2D([0], [0], color="grey", linestyle="--", linewidth=4, label="human")

save_standalone_legend(
    family_handles + [human_handle],
    ncol=1,
    out_path=PLOTS_DIR / "scaling_accuracy_legend.png",
    fontsize=LEGEND_FS,
)

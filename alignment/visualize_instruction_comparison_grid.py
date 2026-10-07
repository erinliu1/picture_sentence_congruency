"""Alignment instruction comparison grid: families in columns, instances by size in rows.

Run directly to save PDF and PNG versions under alignment/plots. Rows represent
size rank within each family, rather than matching parameter counts.
"""

import json
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
FAMILY_NAMES = {
    "qwen": "Qwen3-VL",
    "intern": "InternVL3",
    "llava": "LLaVA-OV",
    "plm": "PLM",
    "glm": "GLM-4.6V",
    "gemma3": "Gemma3",
    "gemma4": "Gemma4",
}
CONDITIONS = [
    ("alignment", "Instruct Model @ critical word", "tab:green", "-"),
    ("alignment_instruction", "Instruct Model @ rating readout", "tab:pink", "--"),
]


def load_ci_plot_df(model_id, folder):
    results_dir = MODELS_DIR / model_id / folder
    plot_df = pd.read_csv(results_dir / "coefficients.csv").sort_values("layer_index")
    plot_df["lower"] = plot_df["coefficient"] - 1.96 * plot_df["se"]
    plot_df["upper"] = plot_df["coefficient"] + 1.96 * plot_df["se"]
    return plot_df


def plot_instruction_comparison_grid(out_name="instruction_comparison_grid.pdf"):
    with (MODELS_DIR / "models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)
    families = {}
    for model_id, info in models_lookup.items():
        if not info["family"].endswith("_base"):
            families.setdefault(info["family"], []).append(model_id)
    for model_ids in families.values():
        model_ids.sort(key=lambda mid: float(re.search(r"[\d.]+", models_lookup[mid]["size"])[0]))
    family_order = [family for family in FAMILY_NAMES if family in families]
    family_order.extend(family for family in families if family not in FAMILY_NAMES)
    n_rows = max(len(model_ids) for model_ids in families.values())
    fig, axes = plt.subplots(
        n_rows, len(family_order), figsize=(5.6 * len(family_order), 3.8 * n_rows),
        squeeze=False,
    )

    for col, family in enumerate(family_order):
        model_ids = families[family]
        for row in range(n_rows):
            ax = axes[row, col]
            if row >= len(model_ids):
                ax.axis("off")
                continue
            model_id = model_ids[row]
            for folder, label, color, linestyle in CONDITIONS:
                try:
                    plot_df = load_ci_plot_df(model_id, folder)
                except FileNotFoundError as error:
                    warnings.warn(f"{model_id}: {label} unavailable: {error.filename}", stacklevel=2)
                    continue
                layers = plot_df["layer_index"]
                span = layers.max() - layers.min()
                normalized_layers = (layers - layers.min()) / span if span else layers * 0
                ax.plot(
                    normalized_layers, plot_df["coefficient"], linewidth=6,
                    linestyle=linestyle, color=color,
                )
                ax.fill_between(
                    normalized_layers, plot_df["lower"], plot_df["upper"],
                    color=color, alpha=0.2,
                )
            ax.axhline(0, linestyle=":", color="gray", linewidth=2)
            ax.legend(
                handles=[Patch(color="none", label=models_lookup[model_id]["size"])],
                loc="lower right", fontsize=30, framealpha=0.9,
                handlelength=0, handletextpad=0,
            )
            ax.set_xlim(0, 1)
            # Preserve raw coefficients and autoscale to include confidence bands.
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            # ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_ylim(-0.2, 2)
            ax.set_yticks([0, 1, 2], ["0", "1", "2"])
            ax.tick_params(axis="both", labelsize=28, width=2, length=9,
                           labelbottom=(row == len(model_ids) - 1), labelleft=True)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.grid(True, linewidth=1, alpha=0.3)
        axes[0, col].set_title(FAMILY_NAMES.get(family, family), fontsize=36,
                               fontweight="bold", pad=16)
        axes[len(model_ids) - 1, col].set_xlabel("Normalized layer", fontsize=32)

    axes[min(1, n_rows - 1), 0].set_ylabel(r"Alignment coefficient ($\beta$)", fontsize=36)
    handles = [Line2D([0], [0], color=color, linewidth=4, linestyle=style, label=label)
               for _, label, color, style in CONDITIONS]
    # Center the shared two-row legend below PLM, as in the decoding grid.
    legend = fig.legend(
        handles=handles, loc="center", ncol=1, fontsize=44,
        bbox_to_anchor=(0.5, 0.5),
        bbox_transform=axes[4, 3].transAxes,
        framealpha=0.9,
    )
    legend.set_zorder(10)
    fig.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTS_DIR / out_name
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    plot_instruction_comparison_grid()

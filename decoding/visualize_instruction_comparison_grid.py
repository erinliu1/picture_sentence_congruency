"""Instruction comparison grid: families in columns, instances by size in rows.

Run directly to save PDF and PNG versions under decoding/plots. Rows represent
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
    ("decoding", "Instruct Model @ critical word", "tab:green", "-"),
    ("decoding_instruction", "Instruct Model @ rating readout", "tab:pink", "--"),
]


def load_ci_plot_df(model_id, folder):
    results_dir = MODELS_DIR / model_id / folder
    accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
    bootstrap_df = pd.read_csv(results_dir / "all_bootstraps.csv")
    bootstrap_ci = (
        bootstrap_df.groupby("layer_index")["all_accuracy"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
    )
    bootstrap_ci.columns = ["layer_index", "lower", "upper"]
    return accuracy_df.merge(
        bootstrap_ci, on="layer_index", validate="one_to_one"
    ).sort_values("layer_index")


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
            missing = False
            for folder, label, color, linestyle in CONDITIONS:
                try:
                    plot_df = load_ci_plot_df(model_id, folder)
                except FileNotFoundError as error:
                    warnings.warn(f"{model_id}: {label} unavailable: {error.filename}", stacklevel=2)
                    missing = True
                    continue
                layers = plot_df["layer_index"]
                span = layers.max() - layers.min()
                normalized_layers = (layers - layers.min()) / span if span else layers * 0
                ax.plot(
                    normalized_layers, plot_df["all_accuracy"], linewidth=6,
                    linestyle=linestyle, color=color,
                )
                ax.fill_between(
                    normalized_layers, plot_df["lower"], plot_df["upper"],
                    color=color, alpha=0.2,
                )
            ax.axhline(0.58, linestyle="--", color="gray", linewidth=3)
            # if missing:
            #     ax.text(0.5, 0.5, "Missing comparison", transform=ax.transAxes,
            #             ha="center", fontsize=22)
            ax.legend(
                handles=[Patch(color="none", label=models_lookup[model_id]["size"])],
                loc="lower right", fontsize=30, framealpha=0.9,
                handlelength=0, handletextpad=0,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0.45, 1.0)
            ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
            ax.set_yticks([0.6, 0.8, 1.0], ["0.6", "0.8", "1"])
            ax.tick_params(axis="both", labelsize=28, width=2, length=9,
                           labelbottom=(row == len(model_ids) - 1), labelleft=True)
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.grid(True, linewidth=1, alpha=0.3)
        axes[0, col].set_title(FAMILY_NAMES.get(family, family), fontsize=36,
                               fontweight="bold", pad=16)
        axes[len(model_ids) - 1, col].set_xlabel("Normalized layer", fontsize=32)

    axes[min(1, n_rows - 1), 0].set_ylabel("Decoding accuracy", fontsize=36)
    handles = [Line2D([0], [0], color=color, linewidth=4, linestyle=style, label=label)
               for _, label, color, style in CONDITIONS]
    # Center the shared two-row legend in row 4, column 4, below PLM.
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

# Combined base-vs-instruct decoding accuracy grid across all three families we have base
# counterparts for. Columns = family (Gemma3, Gemma4, InternVL3), left to right. Rows = model size
# within that family, smallest to largest, top to bottom -- rows are aligned by rank, not by
# matching absolute parameter count across families, since the three families don't share sizes.
# Column headers carry the family name; instruct (tab:green, solid) vs. base (tab:orange, dashed)
# is explained once via a shared legend at the top, so each subplot's own legend only needs to say
# which size it is.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from visualize_base_vs_instruct import (
    MODELS_DIR,
    models_lookup,
    PLOTS_BASE_DIR,
    get_normalized_layers,
    INSTRUCT_COLOR,
    BASE_COLOR,
)

COLUMNS = [
    (
        "Gemma3",
        [
            ("gemma3_4b_base", "gemma3_4b"),
            ("gemma3_12b_base", "gemma3_12b"),
            ("gemma3_27b_base", "gemma3_27b"),
        ],
    ),
    (
        "Gemma4",
        [
            ("gemma4_e2b_base", "gemma4_e2b"),
            ("gemma4_e4b_base", "gemma4_e4b"),
            ("gemma4_12b_base", "gemma4_12b"),
            ("gemma4_26b_a4b_base", "gemma4_26b_a4b"),
            ("gemma4_31b_base", "gemma4_31b"),
        ],
    ),
    (
        "InternVL3",
        [
            ("internvl3_1b_base", "internvl3_1b_instruct"),
            ("internvl3_2b_base", "internvl3_2b_instruct"),
            ("internvl3_8b_base", "internvl3_8b_instruct"),
            ("internvl3_14b_base", "internvl3_14b_instruct"),
            ("internvl3_38b_base", "internvl3_38b_instruct"),
            ("internvl3_78b_base", "internvl3_78b_instruct"),
        ],
    ),
]


def load_ci_plot_df(model_id):
    results_dir = MODELS_DIR / model_id / "decoding"
    accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
    bootstrap_df = pd.read_csv(results_dir / "all_bootstraps.csv")

    bootstrap_ci = (
        bootstrap_df.groupby("layer_index")["all_accuracy"]
        .quantile([0.025, 0.975])
        .unstack()
        .reset_index()
    )
    bootstrap_ci.columns = ["layer_index", "lower", "upper"]
    return accuracy_df.merge(bootstrap_ci, on="layer_index")


X_TICKS = [0, 0.25, 0.5, 0.75, 1]
X_TICK_LABELS = ["0", "0.25", "0.5", "0.75", "1"]
Y_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
Y_TICK_LABELS = ["0.5", "0.6", "0.7", "0.8", "0.9", "1"]
Y_TICKS_SPARSE = [0.6, 0.8, 1.0]
Y_TICK_LABELS_SPARSE = ["0.6", "0.8", "1"]
Y_TICKS_GEMMA3 = [0.6, 0.8, 1.0]
Y_TICK_LABELS_GEMMA3 = ["0.6", "0.8", "1"]


def plot_combined_grid(out_name="base_vs_instruct_grid.pdf"):
    n_rows = max(len(pairs) for _, pairs in COLUMNS)
    n_cols = len(COLUMNS)

    # No sharex/sharey: each subplot needs its own tick-label visibility (only specific rows/
    # columns should show axis ticks), which shared axes would force identical across the group.
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 3.8 * n_rows))

    for col, (family_name, pairs) in enumerate(COLUMNS):
        for row in range(n_rows):
            ax = axes[row, col]

            if row >= len(pairs):
                ax.axis("off")
                continue

            base_model_id, instruct_model_id = pairs[row]
            size_label = models_lookup[instruct_model_id]["size"]

            for model_id, linestyle, color in [
                (instruct_model_id, "-", INSTRUCT_COLOR),
                (base_model_id, "--", BASE_COLOR),
            ]:
                plot_df = load_ci_plot_df(model_id)
                normalized_layers = get_normalized_layers(plot_df["layer_index"])

                ax.plot(
                    normalized_layers,
                    plot_df["all_accuracy"],
                    linewidth=5,
                    linestyle=linestyle,
                    color=color,
                )
                ax.fill_between(
                    normalized_layers,
                    plot_df["lower"],
                    plot_df["upper"],
                    color=color,
                    alpha=0.2,
                )

            ax.axhline(0.58, linestyle="--", color="gray", linewidth=3)

            size_handle = Patch(color="none", label=size_label)
            ax.legend(
                handles=[size_handle],
                loc="lower right",
                fontsize=26,
                frameon=True,
                framealpha=0.9,
                handlelength=0,
                handletextpad=0,
            )

            ax.set_xlim(0, 1)
            ax.set_ylim(0.45, 1.0)
            ax.set_xticks(X_TICKS, X_TICK_LABELS)
            ax.set_yticks(Y_TICKS, Y_TICK_LABELS)
            ax.tick_params(axis="both", labelsize=24, width=2, length=9)
            # hidden by default; re-enabled selectively below for specific rows/columns
            ax.tick_params(labelbottom=False, labelleft=(col == 0))
            for spine in ax.spines.values():
                spine.set_linewidth(3)
            ax.grid(True, linewidth=1, alpha=0.3)

        axes[0, col].set_title(family_name, fontsize=30, fontweight="bold", pad=16)

        # x-axis label + tick labels on the bottom-most *active* row of this column (columns have
        # different numbers of sizes, so this isn't always the last row of the grid).
        bottom_row = len(pairs) - 1
        axes[bottom_row, col].set_xlabel("Normalized layer", fontsize=26)
        axes[bottom_row, col].tick_params(axis="x", labelbottom=True, labelsize=24)

    # Extra y-tick labels (sparse: 0.6/0.8/1) on Gemma4's bottom 2 plots and InternVL3's
    # bottommost plot, for readability away from the leftmost column.
    for row, col in [(3, 1), (4, 1), (5, 2)]:
        ax = axes[row, col]
        ax.set_yticks(Y_TICKS_SPARSE, Y_TICK_LABELS_SPARSE)
        ax.tick_params(axis="y", labelleft=True, labelsize=24)

    # Gemma3's 3 plots use their own sparse y-ticks (0.66/0.8/1) instead of the full set.
    for row in range(len(COLUMNS[0][1])):
        ax = axes[row, 0]
        ax.set_yticks(Y_TICKS_GEMMA3, Y_TICK_LABELS_GEMMA3)
        ax.tick_params(axis="y", labelleft=True, labelsize=24)

    # y-axis label on the second row of the leftmost column only.
    axes[1, 0].set_ylabel("Decoding accuracy", fontsize=32)

    # Instruct-vs-base key, placed in the blank whitespace under Gemma3's 3 plots (column 0,
    # the middle of its unused rows) rather than floating above the whole figure.
    legend_row = 3  # aligned with the 4th row of plots (0-indexed row 3)
    legend_handles = [
        Line2D([0], [0], color=INSTRUCT_COLOR, linewidth=5, linestyle="-"),
        Line2D([0], [0], color=BASE_COLOR, linewidth=5, linestyle="--"),
    ]
    axes[legend_row, 0].legend(
        legend_handles,
        ["Instruct Model\n@ critical word", "Base Model\n@ critical word"],
        loc="center",
        fontsize=28,
        frameon=True,
        framealpha=0.9,
    )

    fig.tight_layout()

    plots_dir = PLOTS_BASE_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / out_name
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_combined_grid()

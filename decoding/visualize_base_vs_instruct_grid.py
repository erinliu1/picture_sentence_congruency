# Multi-panel grid version of visualize_base_vs_instruct.py: one subplot per model size, same
# instruct (tab:green, solid) vs. base (pink, dashed) convention and bootstrap CI bands, on a
# shared y-axis so sizes are directly comparable.

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from visualize_base_vs_instruct import (
    MODELS_DIR,
    models_lookup,
    PLOTS_BASE_DIR,
    get_normalized_layers,
    INSTRUCT_COLOR,
    BASE_COLOR,
)


def plot_grid(pairs, out_name, out_dir, suptitle):
    plots_dir = PLOTS_BASE_DIR / out_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (base_model_id, instruct_model_id) in zip(axes, pairs):
        for model_id, linestyle, color in [
            (instruct_model_id, "-", INSTRUCT_COLOR),
            (base_model_id, "--", BASE_COLOR),
        ]:
            title = models_lookup[model_id]["title"]
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
            plot_df = accuracy_df.merge(bootstrap_ci, on="layer_index")
            normalized_layers = get_normalized_layers(plot_df["layer_index"])

            ax.plot(
                normalized_layers,
                plot_df["all_accuracy"],
                linewidth=3,
                linestyle=linestyle,
                color=color,
                label=title,
            )
            ax.fill_between(
                normalized_layers,
                plot_df["lower"],
                plot_df["upper"],
                color=color,
                alpha=0.2,
            )

        ax.axhline(0.5, linestyle=":", color="gray", linewidth=1.5)
        size = models_lookup[instruct_model_id]["size"]
        ax.set_title(size, fontsize=16)
        ax.set_xlabel("Normalized layer", fontsize=13)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="both", labelsize=11, width=1.5, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        ax.grid(True, linewidth=1, alpha=0.3)
        ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    axes[0].set_ylabel("Decoding accuracy", fontsize=14)
    axes[0].set_ylim(0.45, 1.0)

    fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()

    out_path = plots_dir / out_name
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_grid(
        [
            ("gemma3_4b_base", "gemma3_4b"),
            ("gemma3_12b_base", "gemma3_12b"),
            ("gemma3_27b_base", "gemma3_27b"),
        ],
        "gemma3_all_sizes_base_vs_instruct.pdf",
        out_dir="gemma3_base",
        suptitle="Gemma3: instruct vs. base decoding accuracy",
    )

    plot_grid(
        [
            ("gemma4_e2b_base", "gemma4_e2b"),
            ("gemma4_e4b_base", "gemma4_e4b"),
            ("gemma4_12b_base", "gemma4_12b"),
            ("gemma4_26b_a4b_base", "gemma4_26b_a4b"),
            ("gemma4_31b_base", "gemma4_31b"),
        ],
        "gemma4_all_sizes_base_vs_instruct.pdf",
        out_dir="gemma4_base",
        suptitle="Gemma4: instruct vs. base decoding accuracy",
    )

    plot_grid(
        [
            ("internvl3_1b_base", "internvl3_1b_instruct"),
            ("internvl3_2b_base", "internvl3_2b_instruct"),
            ("internvl3_8b_base", "internvl3_8b_instruct"),
            ("internvl3_14b_base", "internvl3_14b_instruct"),
            ("internvl3_38b_base", "internvl3_38b_instruct"),
            ("internvl3_78b_base", "internvl3_78b_instruct"),
        ],
        "internvl3_all_sizes_base_vs_instruct.pdf",
        out_dir="intern_base",
        suptitle="InternVL3: instruct vs. base decoding accuracy",
    )

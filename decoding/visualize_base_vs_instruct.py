# One-off comparison plot: base (pretrained) vs. instruct decoding accuracy, for models where
# we've repeated the hidden-state + decoding setup on a base counterpart. Includes bootstrap CI
# bands (from all_bootstraps.csv, same 2.5/97.5 percentile convention as lr_visualize.py) now that
# lr_bootstrap has been run for the base models too. Instruct line is tab:green, base line is pink,
# regardless of each model's family color in models_lookup.json (this plot is specifically about
# the instruct/base contrast, not family identity).

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

PLOTS_BASE_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/plots")

INSTRUCT_COLOR = "tab:green"
BASE_COLOR = "tab:orange"


def get_normalized_layers(layers):
    return (layers - layers.min()) / (layers.max() - layers.min())


def plot_base_vs_instruct(base_model_id, instruct_model_id, out_name, out_dir="gemma3_base"):
    plots_dir = PLOTS_BASE_DIR / out_dir
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

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
            linewidth=4,
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

    ax.axhline(0.5, linestyle=":", color="gray", linewidth=2, label="chance (0.50)")

    ax.set_xlabel("Normalized layer", fontsize=20)
    ax.set_ylabel("Decoding accuracy", fontsize=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.45, 1.0)
    ax.tick_params(axis="both", labelsize=16, width=2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, linewidth=1.5, alpha=0.3)
    ax.legend(fontsize=15, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    out_path = plots_dir / out_name
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_base_vs_instruct("gemma3_4b_base", "gemma3_4b", "gemma3_4b_base_vs_instruct.pdf")
    plot_base_vs_instruct("gemma3_12b_base", "gemma3_12b", "gemma3_12b_base_vs_instruct.pdf")
    plot_base_vs_instruct("gemma3_27b_base", "gemma3_27b", "gemma3_27b_base_vs_instruct.pdf")

    plot_base_vs_instruct("gemma4_e2b_base", "gemma4_e2b", "gemma4_e2b_base_vs_instruct.pdf", out_dir="gemma4_base")
    plot_base_vs_instruct("gemma4_e4b_base", "gemma4_e4b", "gemma4_e4b_base_vs_instruct.pdf", out_dir="gemma4_base")
    plot_base_vs_instruct("gemma4_12b_base", "gemma4_12b", "gemma4_12b_base_vs_instruct.pdf", out_dir="gemma4_base")
    plot_base_vs_instruct("gemma4_26b_a4b_base", "gemma4_26b_a4b", "gemma4_26b_a4b_base_vs_instruct.pdf", out_dir="gemma4_base")
    plot_base_vs_instruct("gemma4_31b_base", "gemma4_31b", "gemma4_31b_base_vs_instruct.pdf", out_dir="gemma4_base")

    plot_base_vs_instruct("internvl3_1b_base", "internvl3_1b_instruct", "internvl3_1b_base_vs_instruct.pdf", out_dir="intern_base")
    plot_base_vs_instruct("internvl3_2b_base", "internvl3_2b_instruct", "internvl3_2b_base_vs_instruct.pdf", out_dir="intern_base")
    plot_base_vs_instruct("internvl3_8b_base", "internvl3_8b_instruct", "internvl3_8b_base_vs_instruct.pdf", out_dir="intern_base")
    plot_base_vs_instruct("internvl3_14b_base", "internvl3_14b_instruct", "internvl3_14b_base_vs_instruct.pdf", out_dir="intern_base")
    plot_base_vs_instruct("internvl3_38b_base", "internvl3_38b_instruct", "internvl3_38b_base_vs_instruct.pdf", out_dir="intern_base")
    plot_base_vs_instruct("internvl3_78b_base", "internvl3_78b_instruct", "internvl3_78b_base_vs_instruct.pdf", out_dir="intern_base")

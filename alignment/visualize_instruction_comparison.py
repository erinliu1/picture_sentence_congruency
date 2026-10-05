"""Compare layerwise representation–behavior alignment with and without instructions."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/alignment/plots")


def plot_instruction_comparison(model_id):
    """Save a blue/pink coefficient comparison with pointwise 95% confidence bands."""
    with (MODELS_DIR / "models_lookup.json").open(encoding="utf-8") as file:
        info = json.load(file)[model_id]
        title = info["title"]
    plots_dir = PLOTS_DIR / f"{info['family']}_instruction"

    # Load both conditions before creating a figure, so missing inputs fail cleanly.
    conditions = []
    for folder, label, color, linestyle in [
        ("alignment", "Without instructions", "tab:blue", "-"),
        ("alignment_instruction", "With instructions", "tab:pink", "--"),
    ]:
        df = pd.read_csv(MODELS_DIR / model_id / folder / "coefficients.csv")
        df = df.sort_values("layer_index")
        df["lower"] = df["coefficient"] - 1.96 * df["se"]
        df["upper"] = df["coefficient"] + 1.96 * df["se"]
        conditions.append((df, label, color, linestyle))

    fig, ax = plt.subplots(figsize=(10, 5))
    for df, label, color, linestyle in conditions:
        layers = df["layer_index"]
        span = layers.max() - layers.min()
        normalized_layers = (layers - layers.min()) / span if span else layers * 0
        ax.plot(
            normalized_layers, df["coefficient"],
            linewidth=4, linestyle=linestyle, color=color, label=label,
        )
        ax.fill_between(
            normalized_layers, df["lower"], df["upper"], color=color, alpha=0.2,
        )

    ax.axhline(0, linestyle=":", color="gray", linewidth=2, label="Zero alignment")
    ax.set_title(title, fontsize=20)
    ax.set_xlabel("Normalized layer", fontsize=20)
    ax.set_ylabel(r"Alignment coefficient ($\beta$)", fontsize=20)
    ax.set_xlim(0, 1)
    # Autoscale vertically so negative coefficients and confidence bands remain visible.
    ax.tick_params(axis="both", labelsize=16, width=2, length=8)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.grid(True, linewidth=1.5, alpha=0.3)
    ax.legend(fontsize=15, loc="lower right", framealpha=0.9)
    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / f"{model_id}_instruction_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


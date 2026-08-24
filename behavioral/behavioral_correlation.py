from pathlib import Path

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata
from scipy.stats import linregress

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")

PLOTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

human_df = pd.read_csv(HUMAN_PATH)
human_df = (human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean())

def get_spearman_ci(human_df, vlm_df, n_boot=10000, seed=0):
    df = human_df.merge(vlm_df, on=["item_index", "image_word", "condition"], suffixes=("_human", "_vlm"))

    rho, p = spearmanr(df["rating_human"], df["rating_vlm"])
    items = np.sort(df["item_index"].unique())
    n_items = len(items)
    row_positions_by_item = df.groupby("item_index").indices
    max_group_size = max(len(positions) for positions in row_positions_by_item.values())

    human_vals = df["rating_human"].to_numpy()
    vlm_vals = df["rating_vlm"].to_numpy()

    human_padded = np.full((n_items, max_group_size), np.nan)
    vlm_padded = np.full((n_items, max_group_size), np.nan)
    valid_mask = np.zeros((n_items, max_group_size), dtype=bool)
    for i, item in enumerate(items):
        positions = row_positions_by_item[item]
        human_padded[i, :len(positions)] = human_vals[positions]
        vlm_padded[i, :len(positions)] = vlm_vals[positions]
        valid_mask[i, :len(positions)] = True

    rng = np.random.default_rng(seed)
    boot_item_idx = rng.integers(0, n_items, size=(n_boot, n_items))

    human_boot = human_padded[boot_item_idx].reshape(n_boot, -1)
    vlm_boot = vlm_padded[boot_item_idx].reshape(n_boot, -1)
    mask_boot = valid_mask[boot_item_idx].reshape(n_boot, -1)

    human_ranks = rankdata(np.where(mask_boot, human_boot, -np.inf), axis=1)
    vlm_ranks = rankdata(np.where(mask_boot, vlm_boot, -np.inf), axis=1)

    n_valid = mask_boot.sum(axis=1, keepdims=True)
    human_centered = (human_ranks - (human_ranks * mask_boot).sum(axis=1, keepdims=True) / n_valid) * mask_boot
    vlm_centered = (vlm_ranks - (vlm_ranks * mask_boot).sum(axis=1, keepdims=True) / n_valid) * mask_boot

    covariance = (human_centered * vlm_centered).sum(axis=1)
    human_var = (human_centered ** 2).sum(axis=1)
    vlm_var = (vlm_centered ** 2).sum(axis=1)
    bootstrap_rhos = covariance / np.sqrt(human_var * vlm_var)

    ci = np.percentile(bootstrap_rhos, [2.5, 97.5])

    return rho, p, ci, df


def plot_pairwise(combined_df, model_id, title):
    plt.figure(figsize=(10, 5))

    rng = np.random.default_rng(42)

    # Small jitter for visualization only
    jitter = 0

    colors = {
        "congruent": "tab:green",
        "incongruent": "tab:red"
    }

    for condition, group in combined_df.groupby("condition"):
        plt.scatter(
            group["rating_human"] + rng.uniform(-jitter, jitter, len(group)),
            group["rating_vlm"] + rng.uniform(-jitter, jitter, len(group)),
            color=colors[condition],
            s=200,
            alpha=0.7,
            edgecolors="none",
            label=condition.capitalize()
        )

    # Regression line (computed on original data)
    fit = linregress(combined_df["rating_human"], combined_df["rating_vlm"])
    x = np.linspace(
        combined_df["rating_human"].min(),
        combined_df["rating_human"].max(),
        100
    )

    y = fit.intercept + fit.slope * x

    plt.plot(x, y, color="black", linestyle="--", linewidth=3, label="OLS fit")

    plt.xlabel("Mean Human Rating", fontsize=22)
    plt.ylabel("Mean VLM Rating", fontsize=22)

    plt.xlim(0.9, 5.1)
    plt.ylim(0.9, 5.1)

    plt.xticks([1, 2, 3, 4, 5], fontsize=22)
    plt.yticks([1, 2, 3, 4, 5], fontsize=22)

    plt.legend(loc="best", frameon=True, fontsize=22,)
    plt.tight_layout()
    plt.title(title, fontsize=22)

    ax = plt.gca()

    ax.spines["top"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)

    ax.tick_params(width=2, length=8)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.grid(True, linewidth=2, alpha=0.3)

    family = models_lookup[model_id]["family"]
    plot_dir = PLOTS_DIR / family
    plot_dir.mkdir(exist_ok=True)

    plt.savefig(plot_dir / f"pairwise_{model_id}.pdf", bbox_inches="tight")
    print(f"Pairwise plot for {model_id} saved.")
    plt.close()

def extract_correlation(model_id):
    model_dir = MODELS_DIR / model_id
    title = models_lookup[model_id]["title"]
    try:
        vlm_csv_path = model_dir / "behavior" / "ratings.csv"
        vlm_df = pd.read_csv(vlm_csv_path)
    except:
        return
    rho, p, ci, combined_df = get_spearman_ci(human_df, vlm_df)
    plot_pairwise(combined_df, model_id, title)
    return rho, p, ci

def save_all_correlation():
    records = []
    for model_id in models_lookup.keys():
        result = extract_correlation(model_id)
        if result is None:
            print(f"ratings.csv not found for {models_lookup[model_id]['title']}, skipping")
            records.append({"source": model_id, "spearman_rho": None, "p_value": None, "ci_lower": None, "ci_upper": None})
            continue
        rho, p, ci = result
        records.append({
            "source": model_id,
            "spearman_rho": rho,
            "p_value": p,
            "ci_lower": ci[0],
            "ci_upper": ci[1]
        })
    correlation_df = pd.DataFrame(records)
    correlation_df.to_csv(RESULTS_DIR / "correlation.csv", index=False)
    print("All correlation results saved.")
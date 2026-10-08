"""Correlate behavioral accuracy with final-layer decoding accuracy.

Compute Pearson correlations across instruction-tuned models separately at the
critical-word and rating-readout positions. Missing pairs are omitted per
position; final-layer accuracy comes from the largest layer_index.
"""

import json
import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def behavioral_decoding_correlation():
    with (ROOT_DIR / "models/models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)
    behavioral = pd.read_csv(ROOT_DIR / "behavioral/results/accuracy.csv")
    if behavioral["source"].duplicated().any():
        raise ValueError("Behavioral accuracy contains duplicate model IDs")
    behavioral = behavioral.set_index("source")["accuracy"]

    rows = []
    for position, folder in [
        ("critical_word", "decoding"),
        ("rating_readout", "decoding_instruction"),
    ]:
        pairs = []
        for model_id, info in models_lookup.items():
            if info["family"].endswith("_base"):
                continue
            path = ROOT_DIR / "models" / model_id / folder / "layerwise_accuracy.csv"
            if model_id not in behavioral.index or not path.exists():
                warnings.warn(f"Missing {position} correlation inputs for {model_id}", stacklevel=2)
                continue
            accuracy = pd.read_csv(path).sort_values("layer_index")
            if accuracy.empty or accuracy["layer_index"].isna().any():
                raise ValueError(f"Empty or invalid layer indices: {path}")
            if accuracy["layer_index"].duplicated().any():
                raise ValueError(f"Duplicate layer indices: {path}")
            pairs.append((behavioral.loc[model_id], accuracy["all_accuracy"].iloc[-1]))

        paired = pd.DataFrame(pairs, columns=["behavioral_accuracy", "final_layer_accuracy"]).dropna()
        if len(paired) < 3:
            raise ValueError(f"At least three complete models are required for {position}")
        if (paired.nunique() < 2).any():
            raise ValueError(f"Correlation is undefined for constant {position} inputs")
        result = pearsonr(
            paired["behavioral_accuracy"], paired["final_layer_accuracy"],
            alternative="two-sided",
        )
        rows.append({
            "position": position,
            "n_models": len(paired),
            "pearson_r": result.statistic,
            "two_sided_p": result.pvalue,
        })

    out_path = RESULTS_DIR / "behavioral_decoding_correlation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return out_path


if __name__ == "__main__":
    behavioral_decoding_correlation()

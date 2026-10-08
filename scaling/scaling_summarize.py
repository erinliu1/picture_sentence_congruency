"""Combine behavioral, decoding, and alignment metrics per instruct model.

params_b follows decoding/visualize_scaling.py: the first number in the size
label (total count for MoE, effective count for E2B/E4B). log_parameters is
log10(params_b), matching that plot's base-10 logarithmic axis. Peak metrics
are raw maxima; final metrics use the largest layer_index. Missing inputs
remain blank. Base models are excluded.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
COLUMNS = [
    "model_id", "model_family", "params_b", "log_parameters", "behavioral_accuracy",
    "peak_CW_decoding", "final_CW_decoding", "peak_RR_decoding", "final_RR_decoding",
    "peak_CW_alignment", "final_CW_alignment", "peak_RR_alignment", "final_RR_alignment",
]


def peak_and_final(path, value_column):
    if not path.exists():
        return None, None
    df = pd.read_csv(path).sort_values("layer_index")
    if df.empty or df[["layer_index", value_column]].isna().any().any():
        raise ValueError(f"Empty or incomplete results: {path}")
    if df["layer_index"].duplicated().any():
        raise ValueError(f"Duplicate layer indices: {path}")
    return df[value_column].max(), df[value_column].iloc[-1]


def scaling_summarize():
    with (ROOT_DIR / "models/models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)
    behavioral = pd.read_csv(ROOT_DIR / "behavioral/results/accuracy.csv")
    if behavioral["source"].duplicated().any():
        raise ValueError("Behavioral accuracy contains duplicate model IDs")
    behavioral = behavioral.set_index("source")["accuracy"]

    rows = []
    for model_id, info in models_lookup.items():
        if info["family"].endswith("_base"):
            continue
        params_b = float(re.search(r"(\d+(?:\.\d+)?)", info["size"]).group(1))
        row = {
            "model_id": model_id,
            "model_family": info["family"],
            "params_b": params_b,
            "log_parameters": np.log10(params_b),
            "behavioral_accuracy": behavioral.get(model_id, None),
        }
        for metric, filename, value_column in [
            ("decoding", "layerwise_accuracy.csv", "all_accuracy"),
            ("alignment", "coefficients.csv", "coefficient"),
        ]:
            for position, suffix in [("CW", ""), ("RR", "_instruction")]:
                path = ROOT_DIR / "models" / model_id / f"{metric}{suffix}" / filename
                peak, final = peak_and_final(path, value_column)
                row[f"peak_{position}_{metric}"] = peak
                row[f"final_{position}_{metric}"] = final
        rows.append(row)

    summary = pd.DataFrame(rows, columns=COLUMNS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "scaling_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved {len(summary)} model rows to {out_path}")
    return out_path


if __name__ == "__main__":
    scaling_summarize()

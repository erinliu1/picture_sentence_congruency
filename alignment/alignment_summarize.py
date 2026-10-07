"""Export peak and final-layer alignment coefficients for every model/condition.

Depth is normalized as (layer - first layer) / (last layer - first layer),
matching the visualization scripts. If several layers share the peak coefficient,
the earliest layer is reported. Missing result files produce a row with blank
metrics. Base models currently have no alignment results.
"""

import json
import warnings
from pathlib import Path

import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def alignment_summarize():
    with (MODELS_DIR / "models_lookup.json").open(encoding="utf-8") as file:
        models_lookup = json.load(file)

    rows = []
    for model_id, info in models_lookup.items():
        is_base = info["family"].endswith("_base")
        if is_base:
            conditions = [("alignment", "critical_word")]
        else:
            conditions = [
                ("alignment", "critical_word"),
                ("alignment_instruction", "rating_readout"),
            ]
        for folder, condition in conditions:
            row = {
                "model_id": model_id,
                "model_type": "base" if is_base else "instruct",
                "decoding": condition,
                "peak_coeff": None,
                "peak_normalized_layer": None,
                "final_layer_coeff": None,
            }
            result_path = MODELS_DIR / model_id / folder / "coefficients.csv"
            if not result_path.exists():
                if not is_base:
                    warnings.warn(f"Missing results: {result_path}", stacklevel=2)
                rows.append(row)
                continue
            df = pd.read_csv(result_path).sort_values("layer_index").reset_index(drop=True)
            if df.empty or df[["layer_index", "coefficient"]].isna().any().any():
                raise ValueError(f"Empty or incomplete results: {result_path}")
            if df["layer_index"].duplicated().any():
                raise ValueError(f"Duplicate layer indices: {result_path}")
            peak = df.loc[df["coefficient"].idxmax()]
            final = df.iloc[-1]
            first_layer = df["layer_index"].iloc[0]
            span = final["layer_index"] - first_layer
            row.update({
                "peak_coeff": peak["coefficient"],
                "peak_normalized_layer": (peak["layer_index"] - first_layer) / span if span else 0.0,
                "final_layer_coeff": final["coefficient"],
            })
            rows.append(row)

    summary = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "alignment_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"Saved {len(summary)} model/condition rows to {out_path}")

    return out_path


if __name__ == "__main__":
    alignment_summarize()

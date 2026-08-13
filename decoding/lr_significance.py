from pathlib import Path

import json
import pandas as pd
import numpy as np

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

def lr_significance(model_id):
    models_dir = MODELS_DIR / model_id
    
    results_dir = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/decoding")
    output_path = results_dir / "layerwise_significance.csv"

    if output_path.exists():
        print(f"Layerwise significance results already exist for {model_id}, skipping significance testing.")
        return
        
    observed_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
    max_stats_df = pd.read_csv(results_dir / "max_stats.csv")

    null = max_stats_df["max_accuracy"].to_numpy()
    threshold_95 = np.quantile(null, 0.95)

    p_values, significant = [], []
    for _, row in observed_df.iterrows():
        obs = row["all_accuracy"]
        p = (1 + np.sum(null >= obs)) / (len(null) + 1)
        p_values.append(p)
        significant.append(obs > threshold_95)

    observed_df["threshold_95"] = threshold_95
    observed_df["maxstat_p"] = p_values
    observed_df["exceeds_95th_percentile"] = significant


    observed_df[
        [
            "layer_index",
            "all_accuracy",
            "threshold_95",
            "maxstat_p",
            "exceeds_95th_percentile",
        ]
    ].to_csv(output_path, index=False)
    print(f"Layerwise significance results saved to {output_path}")
from pathlib import Path
import json
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/results")
RESULTS_DIR.mkdir(exist_ok=True)


def lr_significance_summary():
    summary_data = []
    for model_id in models_lookup.keys():
        decoding_dir = MODELS_DIR / model_id / "decoding"
        sig_path = decoding_dir / "layerwise_significance.csv"
        try:
            sig_df = pd.read_csv(sig_path)
        except FileNotFoundError:
            print(f"No layerwise_significance.csv for {model_id}, skipping.")
            continue

        total_layers = len(sig_df)
        significant_mask = sig_df["exceeds_95th_percentile"].astype(bool)
        n_significant = int(significant_mask.sum())
        non_significant_layers = sig_df.loc[~significant_mask, "layer_index"].tolist()
        proportion_significant = n_significant / total_layers if total_layers else float("nan")

        summary_data.append({
            "source": model_id,
            "n_significant_layers": n_significant,
            "total_layers": total_layers,
            "non_significant_layers": ";".join(str(l) for l in non_significant_layers),
            "proportion_significant": proportion_significant,
        })

    summary_df = pd.DataFrame(summary_data)
    output_path = RESULTS_DIR / "layerwise_significance_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Layerwise significance summary saved to {output_path}")


if __name__ == "__main__":
    lr_significance_summary()

from pathlib import Path
import pandas as pd
import json

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/results")
RESULTS_DIR.mkdir(exist_ok=True)

def lr_summarize():
    summary_data = []
    for model_id in models_lookup.keys():
        results_dir = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/decoding")
        main_accuracy_df = pd.read_csv(results_dir / "layerwise_accuracy.csv")
        peak_accuracy = main_accuracy_df["all_accuracy"].max()
        peak_layer_index = main_accuracy_df["all_accuracy"].idxmax()
        last_layer_accuracy = main_accuracy_df["all_accuracy"].iloc[-1]

        summary_data.append({
            "source": model_id,
            "peak_decoding_accuracy": peak_accuracy,
            "peak_layer_index": peak_layer_index,
            "total_layers": len(main_accuracy_df),
            "last_layer_decoding_accuracy": last_layer_accuracy,
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(f"Summary saved to {RESULTS_DIR / 'summary.csv'}")


    
    
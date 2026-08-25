from alignment_rep_bev import get_alignment
from alignment_visualize import plot_alignment, plot_alignment_for_all_models

from pathlib import Path
import pandas as pd
import json

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

ALIGNMENT_DIR = Path("/Intern/Erin/picture_sentence_congruency/alignment")
RESULTS_DIR = ALIGNMENT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

all_families = set()
all_results = []
for model_id in models_lookup.keys():
    df = get_alignment(model_id)
    max_coeff = df["coefficient"].max()
    max_layer_index = df["coefficient"].idxmax()
    total_layers = len(df)
    last_layer_coeff = df["coefficient"].iloc[-1]
    all_results.append({
        "source": model_id,
        "max_coeff": max_coeff,
        "max_layer_index": max_layer_index,
        "total_layers": total_layers,
        "last_layer_coeff": last_layer_coeff
    })
    plot_alignment(model_id)
    all_families.add(models_lookup[model_id]["family"])

save_df = pd.DataFrame(all_results)
save_df.to_csv(RESULTS_DIR / "alignment_summary.csv", index=False)
for family in all_families:
    plot_alignment_for_all_models(family)

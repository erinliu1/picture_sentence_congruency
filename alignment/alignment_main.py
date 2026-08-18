from alignment_rep_bev import get_alignment
from alignment_visualize import plot_alignment, plot_alignment_for_all_models

from pathlib import Path
import json
MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

for model_id in models_lookup.keys():
    if models_lookup[model_id]["family"] == "qwen":
        get_alignment(model_id)
        plot_alignment(model_id)
        print('✅')

plot_alignment_for_all_models('qwen')
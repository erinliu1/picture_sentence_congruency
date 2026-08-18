from hs_glm import extract_hidden_states

import json
from pathlib import Path

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)


for model_id in models_lookup.keys():
    if models_lookup[model_id]["family"] == "glm":
        extract_hidden_states(model_id)
        print('✅')


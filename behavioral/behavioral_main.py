# from behavioral_qwen import extract_ratings
# from behavioral_intern import extract_ratings
from behavioral_plm import extract_ratings
from behavioral_accuracy import extract_accuracy_VLM, save_all_accuracy
from behavioral_correlation import extract_correlation, save_all_correlation
from behavioral_health import extract_health, save_all_health

import json
from pathlib import Path

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

# model_id = "plm_8b"
# extract_ratings(model_id)
# extract_accuracy_VLM(model_id)
# extract_correlation(model_id)
# extract_health(model_id)
for model_id in ["plm_1b", "plm_3b", "plm_8b"]:
    try:
        extract_ratings(model_id)
        extract_accuracy_VLM(model_id)
        extract_correlation(model_id)
        extract_health(model_id)
    except Exception as e:
        print(f"Error extracting ratings for {model_id}: {e}")
        continue

save_all_accuracy()
save_all_correlation()
save_all_health()
from lr_train import lr_train
from lr_bootstrap import lr_bootstrap
from lr_permutation import run_all_permutations
from lr_significance import lr_significance
from lr_visualize import lr_visualize, lr_visualize_all_models

from pathlib import Path
import json
MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

model_id = "internvl3_78b_instruct"
# for model_id in models_lookup.keys():
#     if models_lookup[model_id]["family"] == "intern":
#         lr_train(model_id)
#         lr_bootstrap(model_id)
run_all_permutations(model_id, n_permutations=1000)
lr_significance(model_id)
lr_visualize(model_id)

lr_visualize_all_models("intern")
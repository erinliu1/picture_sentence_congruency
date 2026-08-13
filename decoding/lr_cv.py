import json
from pathlib import Path

from sklearn.model_selection import KFold

STIMULI_PATH = "/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json"
CV_DIR = Path("/Intern/Erin/picture_sentence_congruency/decoding/cv")
CV_DIR.mkdir(exist_ok=True)

SEED = 42
N_SPLITS = 10
N_ITEMS = 80  # stimulus sets 0-79, same count used throughout the pipeline

# ============================================================
# 1. Cross-validation fold split -- which items train/test per fold
# ============================================================

item_indices = list(range(N_ITEMS))
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

cv_indices = {}
for fold_index, (train_indices, test_indices) in enumerate(kf.split(item_indices)):
    fold_index = fold_index + 1
    cv_indices[fold_index] = {
        "train": train_indices.tolist(),
        "test": test_indices.tolist(),
    }

with open(CV_DIR / "cv_indices.json", "w", encoding="utf-8") as f:
    json.dump(cv_indices, f, indent=4)

# ============================================================
# 2. Item -> hidden-state .pt filenames lookup
# ============================================================

with open(STIMULI_PATH, "r") as f:
    sentences = json.load(f)

item_lookup = {}
for item_index, item in enumerate(sentences):
    word_a, word_b = item['word_options']

    congruent_pairs = [(word_a, word_a), (word_b, word_b)]
    incongruent_pairs = [(word_a, word_b), (word_b, word_a)]

    item_lookup[item_index] = []
    for image_word, final_word in congruent_pairs + incongruent_pairs:
        is_congruent = (image_word == final_word)
        condition = 'congruent' if is_congruent else 'incongruent'
        item_key = f"{item_index}_{image_word}_{condition}.pt"
        item_lookup[item_index].append(item_key)

with open(CV_DIR / "item_lookup.json", "w") as f:
    json.dump(item_lookup, f)

# ============================================================
# 3. Join: per-fold train/test .pt filename lists
# ============================================================

cv_lookup_pt = {}
for fold_index, indices in cv_indices.items():
    cv_lookup_pt[fold_index] = {
        "train": {},
        "test": {},
    }
    for item_index in indices["train"]:
        cv_lookup_pt[fold_index]["train"][item_index] = item_lookup[item_index]
    for item_index in indices["test"]:
        cv_lookup_pt[fold_index]["test"][item_index] = item_lookup[item_index]

# print(cv_lookup_pt.keys()) -> dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# print(cv_lookup_pt[1].keys()) -> dict_keys(['train', 'test'])
# print(cv_lookup_pt[1]["train"].keys()) -> dict_keys([1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26, 27, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79])
# print(cv_lookup_pt[1]["train"][1]) -> ['1_disappointment_congruent.pt', '1_delight_congruent.pt', '1_disappointment_incongruent.pt', '1_delight_incongruent.pt']

with open(CV_DIR / "cv_lookup_pt.json", "w") as f:
    json.dump(cv_lookup_pt, f)

import json
import pandas as pd
from pathlib import Path

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

BEHAVIORAL_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral")
RESULTS_DIR = BEHAVIORAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

def compute_concordance(df):
    # for every stimulus set (item_index), compare the two incongruent ratings against the two congruent
    # ratings: incongruent_1 < congruent_1, incongruent_1 < congruent_2, incongruent_2 < congruent_1,
    # incongruent_2 < congruent_2. take the proportion of these four comparisons that hold, then average
    # that proportion across all stimulus sets.
    item_scores = []
    for item_index, group in df.groupby("item_index"):
        congruent_ratings = group.loc[group["condition"] == "congruent", "rating"].tolist()
        incongruent_ratings = group.loc[group["condition"] == "incongruent", "rating"].tolist()
        comparisons = [
            incongruent_rating < congruent_rating
            for incongruent_rating in incongruent_ratings
            for congruent_rating in congruent_ratings
        ]
        item_scores.append(sum(comparisons) / len(comparisons))
    return sum(item_scores) / len(item_scores)

def extract_concordance(model_id):
    try:
        ratings = pd.read_csv(MODELS_DIR / model_id / "behavior" / "ratings.csv")
    except FileNotFoundError:
        return None
    return compute_concordance(ratings)

def save_all_concordance():
    concordance_scores = []
    for model_id in models_lookup.keys():
        concordance = extract_concordance(model_id)
        if concordance is None:
            print(f"ratings.csv not found for {models_lookup[model_id]['title']}, skipping")
            concordance_scores.append({"source": model_id, "concordance": None})
            continue
        concordance_scores.append({
            "source": model_id,
            "concordance": concordance
        })

    df = pd.DataFrame(concordance_scores)
    df.to_csv(RESULTS_DIR / "concordance_scores.csv", index=False)
    print("All concordance results saved.")

# save_all_concordance()
concordance_df = pd.read_csv(RESULTS_DIR / "concordance_scores.csv")
correlation_df = pd.read_csv(RESULTS_DIR / "correlation.csv")
# merge
df = pd.merge(concordance_df, correlation_df, on="source")

print(df)

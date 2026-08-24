from pathlib import Path

import json
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
RESULTS_DIR.mkdir(exist_ok=True)

def extract_instruction(model_id):
    try:
        ratings_df = pd.read_csv(MODELS_DIR / model_id / "behavior" / "ratings.csv")
    except:
        print(f"ratings.csv not found for {models_lookup[model_id]['title']}, skipping")
        return

    return {
        "source": model_id,
        "mean_total_probability_mass": ratings_df["total_probability_mass"].mean(),
        "min_total_probability_mass": ratings_df["total_probability_mass"].min(),
    }

def save_all_instruction():
    instruction_records = []
    for model_id in models_lookup.keys():
        record = extract_instruction(model_id)
        if record is not None:
            instruction_records.append(record)

    instruction_df = pd.DataFrame(instruction_records)
    instruction_df.to_csv(RESULTS_DIR / "instruction_following.csv", index=False)
    print("All instruction-following results saved.")


# save_all_instruction()
df = pd.read_csv(RESULTS_DIR / "instruction_following.csv").set_index("source")
df = df[
    ~df.index.map(
        lambda model_id: models_lookup[model_id]["family"] in ["plm", "llava"]
    )
]
print(df['min_total_probability_mass'].min())
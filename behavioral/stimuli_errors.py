from pathlib import Path

import json
import matplotlib.pyplot as plt
import pandas as pd

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

STIMULI_PATH = Path("/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json")
with open(STIMULI_PATH, "r", encoding="utf-8") as file:
    sentences = json.load(file)

HUMAN_PATH = Path("/Intern/Erin/picture_sentence_congruency/survey/results/ratings.csv")
RESULTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/results")
RESULTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("/Intern/Erin/picture_sentence_congruency/behavioral/plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Exclude models with known problems
EXCLUDED_MODELS = {
    model_id for model_id, info in models_lookup.items() if model_id == "llava_ov_05b"
    or (info["family"] == "plm" and info["size"] in ("1B", "8B"))
    or (info["family"] == "gemma4" and info["type"] == "PLE")
    or (info["family"] == "intern" and info["size"] in ("1B", "2B"))
}
INCLUDED_MODELS = [model_id for model_id in models_lookup if model_id not in EXCLUDED_MODELS]

def get_final_word(item_index, image_word, condition):
    # congruent trials show the image_word itself as the sentence's final word;
    # incongruent trials show the *other* word_option instead (see behavioral_qwen.py's trial construction).
    word_a, word_b = sentences[item_index]["word_options"]
    if condition == "congruent":
        return image_word
    return word_b if image_word == word_a else word_a


def get_sentence(item_index, image_word, condition):
    frame = sentences[item_index]["sentence_frame"]
    final_word = get_final_word(item_index, image_word, condition)
    return f"{frame} {final_word}."


def save_stimuli_errors():
    mistake_frames = []
    for model_id in INCLUDED_MODELS:
        try:
            ratings_df = pd.read_csv(MODELS_DIR / model_id / "behavior" / "ratings.csv")
        except FileNotFoundError:
            print(f"ratings.csv not found for {models_lookup[model_id]['title']}, skipping")
            continue

        ratings_df = ratings_df[["item_index", "image_word", "condition", "rating"]].copy()
        ratings_df["mistake"] = (ratings_df["rating"] > 3) != (ratings_df["condition"] == "congruent")
        ratings_df["model_id"] = model_id
        mistake_frames.append(ratings_df[["item_index", "image_word", "condition", "mistake", "model_id"]])

    all_mistakes_df = pd.concat(mistake_frames, ignore_index=True)
    mistake_counts_df = (
        all_mistakes_df.groupby(["item_index", "image_word", "condition"], as_index=False)["mistake"]
        .sum()
        .rename(columns={"mistake": "num_model_mistakes"})
    )
    mistake_counts_df["num_model_mistakes"] = mistake_counts_df["num_model_mistakes"].astype(int)

    def list_models_with_mistake(group):
        wrong_model_ids = group.loc[group["mistake"], "model_id"]
        return "; ".join(models_lookup[model_id]["title"] for model_id in wrong_model_ids)

    models_with_mistake_df = (
        all_mistakes_df.groupby(["item_index", "image_word", "condition"])
        .apply(list_models_with_mistake, include_groups=False)
        .reset_index(name="models_with_mistake")
    )
    mistake_counts_df = mistake_counts_df.merge(
        models_with_mistake_df, on=["item_index", "image_word", "condition"]
    )

    human_df = pd.read_csv(HUMAN_PATH)
    human_mean_df = human_df.groupby(["item_index", "image_word", "condition"], as_index=False)["rating"].mean()
    human_mean_df["human_mistake"] = (human_mean_df["rating"] > 3) != (human_mean_df["condition"] == "congruent")

    errors_df = mistake_counts_df.merge(
        human_mean_df[["item_index", "image_word", "condition", "human_mistake"]],
        on=["item_index", "image_word", "condition"],
        how="outer",
    )
    errors_df["num_model_mistakes"] = errors_df["num_model_mistakes"].fillna(0).astype(int)
    errors_df["human_mistake"] = errors_df["human_mistake"].fillna(False)

    # keep only stimuli where at least one included model, or the humans, got it wrong
    errors_df = errors_df[(errors_df["num_model_mistakes"] > 0) | errors_df["human_mistake"]].copy()

    errors_df["sentence"] = errors_df.apply(
        lambda row: get_sentence(row["item_index"], row["image_word"], row["condition"]), axis=1
    )

    errors_df = errors_df[["item_index", "sentence", "image_word", "condition", "num_model_mistakes", "models_with_mistake", "human_mistake"]]
    errors_df = errors_df.sort_values("num_model_mistakes", ascending=False).reset_index(drop=True)

    errors_df.to_csv(RESULTS_DIR / "stimuli_errors.csv", index=False)
    print(f"Stimuli errors saved ({len(errors_df)} rows, {len(INCLUDED_MODELS)} models included).")



save_stimuli_errors()
df = pd.read_csv(RESULTS_DIR / "stimuli_errors.csv")
# out of all the stimuli where exactly 1 model got it wrong, how many belonged to each model? print the counts of model_id
# single_mistake_df = df[df["num_model_mistakes"] == 1]
# print(single_mistake_df["models_with_mistake"].value_counts())

# plot the distribution of num_model_mistakes across stimuli
# fig, ax = plt.subplots(figsize=(8, 5))
# max_mistakes = df["num_model_mistakes"].max()
# ax.hist(df["num_model_mistakes"], bins=range(0, max_mistakes + 2), align="left", edgecolor="black")
# ax.set_xticks(range(0, max_mistakes + 1))
# ax.set_xlabel("Number of models that made a mistake")
# ax.set_ylabel("Number of stimuli")
# fig.tight_layout()
# fig.savefig(PLOTS_DIR / "stimuli_errors_distribution.pdf")
# plt.show()

# filter out the ones that only humans got wrong, and then print the total number of rows remaining
# model_mistake_df = df[df["num_model_mistakes"] > 0]
# print(len(model_mistake_df)) 

# how many stimuli tripped up at least half of the models?
half_models = len(INCLUDED_MODELS) / 2
print(half_models)
tripped_half_df = df[df["num_model_mistakes"] >= half_models]
print(len(tripped_half_df))



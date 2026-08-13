from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

STIMULI_PATH = "/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json"
with open(STIMULI_PATH, "r") as f:
    sentences = json.load(f)

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 2026

OUTPUT_DIRECTORY = Path("/Intern/Erin/picture_sentence_congruency/survey/survey_assignments")
COMBINED_OUTPUT_PATH = OUTPUT_DIRECTORY / "all_surveys.json"

# Image files are assumed to be named after the image word.
IMAGE_EXTENSION = ".png"

SURVEY_NAMES = ("A", "B", "C", "D")


# ============================================================
# HELPERS
# ============================================================

def validate_sentences(items: list[dict[str, Any]]) -> None:
    """Validate the expected sentences.py structure."""
    if len(items) == 0:
        raise ValueError("The sentences list is empty.")

    if len(items) % 4 != 0:
        raise ValueError(
            f"Expected the number of stimulus sets to be divisible by 4, "
            f"but found {len(items)}."
        )

    for item_index, item in enumerate(items):
        if "sentence_frame" not in item:
            raise ValueError(
                f"Stimulus set {item_index} is missing 'sentence_frame'."
            )

        if "word_options" not in item:
            raise ValueError(
                f"Stimulus set {item_index} is missing 'word_options'."
            )

        word_options = item["word_options"]

        if not isinstance(word_options, list) or len(word_options) != 2:
            raise ValueError(
                f"Stimulus set {item_index} must contain exactly two "
                f"'word_options', but found: {word_options!r}"
            )

        if word_options[0] == word_options[1]:
            raise ValueError(
                f"Stimulus set {item_index} has duplicate word options."
            )


def join_sentence(sentence_frame: str, final_word: str) -> str:
    """Join a sentence frame and final word with sensible spacing."""
    sentence_frame = sentence_frame.rstrip()
    final_word = final_word.strip()

    sentence = f"{sentence_frame} {final_word}"

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


def make_sample(
    *,
    item_index: int,
    sentence_frame: str,
    final_word: str,
    image_word: str,
) -> dict[str, Any]:
    """Create one survey sample."""
    congruency = "congruent" if final_word == image_word else "incongruent"

    return {
        "item_index": item_index,
        "sentence_frame": sentence_frame,
        "final_word": final_word,
        "sentence": join_sentence(sentence_frame, final_word),
        "image_word": image_word,
        "image_filename": f"{image_word}{IMAGE_EXTENSION}",
        "congruency": congruency,
    }


def make_four_conditions(
    item_index: int,
    item: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Create the four possible samples for one stimulus set.

    With word options [word_0, word_1], the four samples are:

    0. word_0 + image_0: congruent
    1. word_1 + image_1: congruent
    2. word_0 + image_1: incongruent
    3. word_1 + image_0: incongruent
    """
    sentence_frame = item["sentence_frame"]
    word_0, word_1 = item["word_options"]

    return [
        make_sample(
            item_index=item_index,
            sentence_frame=sentence_frame,
            final_word=word_0,
            image_word=word_0,
        ),
        make_sample(
            item_index=item_index,
            sentence_frame=sentence_frame,
            final_word=word_1,
            image_word=word_1,
        ),
        make_sample(
            item_index=item_index,
            sentence_frame=sentence_frame,
            final_word=word_0,
            image_word=word_1,
        ),
        make_sample(
            item_index=item_index,
            sentence_frame=sentence_frame,
            final_word=word_1,
            image_word=word_0,
        ),
    ]


def generate_surveys(
    items: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    """
    Generate four counterbalanced surveys.

    Every stimulus set contributes exactly one sample to each survey.
    Across the four surveys, all four possible samples from every set
    appear exactly once.

    Each survey contains:
      - one sample from every stimulus set;
      - 50% congruent samples;
      - 50% incongruent samples;
      - a randomized but reproducible question order.
    """
    rng = random.Random(seed)

    number_of_items = len(items)

    # Each offset occurs equally often. With 80 sets, each offset occurs
    # 20 times. Because two offsets are congruent and two are incongruent,
    # every survey receives exactly 40 congruent and 40 incongruent samples.
    offsets = [0, 1, 2, 3] * (number_of_items // 4)
    rng.shuffle(offsets)

    surveys: dict[str, list[dict[str, Any]]] = {
        survey_name: [] for survey_name in SURVEY_NAMES
    }

    for item_index, (item, base_offset) in enumerate(zip(items, offsets)):
        conditions = make_four_conditions(item_index, item)

        for survey_index, survey_name in enumerate(SURVEY_NAMES):
            condition_index = (base_offset + survey_index) % 4
            sample = conditions[condition_index].copy()
            sample["survey"] = survey_name
            surveys[survey_name].append(sample)

    # Give each survey its own fixed randomized order.
    for survey_index, survey_name in enumerate(SURVEY_NAMES):
        survey_rng = random.Random(seed + 1000 + survey_index)
        survey_rng.shuffle(surveys[survey_name])

        for question_number, sample in enumerate(
            surveys[survey_name],
            start=1,
        ):
            sample["question_number"] = question_number

    return surveys


def verify_surveys(
    surveys: dict[str, list[dict[str, Any]]],
    number_of_items: int,
) -> None:
    """Verify balancing and counterbalancing constraints."""
    expected_per_condition = number_of_items // 4

    for survey_name, samples in surveys.items():
        if len(samples) != number_of_items:
            raise ValueError(
                f"Survey {survey_name} contains {len(samples)} questions; "
                f"expected {number_of_items}."
            )

        item_indices = [sample["item_index"] for sample in samples]

        if len(set(item_indices)) != number_of_items:
            raise ValueError(
                f"Survey {survey_name} does not contain exactly one sample "
                f"from every stimulus set."
            )

        congruent_count = sum(
            sample["congruency"] == "congruent" for sample in samples
        )
        incongruent_count = sum(
            sample["congruency"] == "incongruent" for sample in samples
        )

        expected_half = number_of_items // 2

        if congruent_count != expected_half:
            raise ValueError(
                f"Survey {survey_name} has {congruent_count} congruent "
                f"samples; expected {expected_half}."
            )

        if incongruent_count != expected_half:
            raise ValueError(
                f"Survey {survey_name} has {incongruent_count} incongruent "
                f"samples; expected {expected_half}."
            )

    # Verify that each stimulus set appears in all four possible conditions
    # exactly once across Surveys A-D.
    for item_index in range(number_of_items):
        observed_pairs = set()

        for survey_name in SURVEY_NAMES:
            sample = next(
                sample
                for sample in surveys[survey_name]
                if sample["item_index"] == item_index
            )
            observed_pairs.add(
                (sample["final_word"], sample["image_word"])
            )

        if len(observed_pairs) != 4:
            raise ValueError(
                f"Stimulus set {item_index} does not cover four unique "
                f"word-image combinations across the four surveys."
            )

    # Verify equal use of each of the four condition types in each survey.
    # This is stronger than merely checking 40 congruent / 40 incongruent.
    for survey_name, samples in surveys.items():
        condition_counts: dict[tuple[int, int], int] = {}

        for sample in samples:
            original_item = sentences[sample["item_index"]]
            word_0, word_1 = original_item["word_options"]

            word_index = 0 if sample["final_word"] == word_0 else 1
            image_index = 0 if sample["image_word"] == word_0 else 1
            key = (word_index, image_index)

            condition_counts[key] = condition_counts.get(key, 0) + 1

        expected_counts = {
            (0, 0): expected_per_condition,
            (1, 1): expected_per_condition,
            (0, 1): expected_per_condition,
            (1, 0): expected_per_condition,
        }

        if condition_counts != expected_counts:
            raise ValueError(
                f"Survey {survey_name} condition counts are unbalanced: "
                f"{condition_counts}"
            )


def save_surveys(
    surveys: dict[str, list[dict[str, Any]]],
) -> None:
    """Save one JSON file per survey and one combined JSON file."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    combined_output = {
        "random_seed": RANDOM_SEED,
        "number_of_surveys": len(SURVEY_NAMES),
        "number_of_questions_per_survey": len(next(iter(surveys.values()))),
        "surveys": surveys,
    }

    with COMBINED_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(combined_output, file, indent=2, ensure_ascii=False)
        file.write("\n")

    for survey_name, samples in surveys.items():
        output_path = OUTPUT_DIRECTORY / f"survey_{survey_name}.json"

        output = {
            "survey": survey_name,
            "random_seed": RANDOM_SEED,
            "number_of_questions": len(samples),
            "questions": samples,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(output, file, indent=2, ensure_ascii=False)
            file.write("\n")


def print_summary(
    surveys: dict[str, list[dict[str, Any]]],
) -> None:
    """Print a concise balance summary."""
    print("\nGenerated surveys:\n")

    for survey_name, samples in surveys.items():
        congruent_count = sum(
            sample["congruency"] == "congruent" for sample in samples
        )
        incongruent_count = sum(
            sample["congruency"] == "incongruent" for sample in samples
        )

        print(
            f"Survey {survey_name}: "
            f"{len(samples)} questions | "
            f"{congruent_count} congruent | "
            f"{incongruent_count} incongruent"
        )

    print(f"\nSaved files to: {OUTPUT_DIRECTORY.resolve()}")


def main() -> None:
    validate_sentences(sentences)

    surveys = generate_surveys(
        items=sentences,
        seed=RANDOM_SEED,
    )

    verify_surveys(
        surveys=surveys,
        number_of_items=len(sentences),
    )

    save_surveys(surveys)
    print_summary(surveys)


if __name__ == "__main__":
    main()

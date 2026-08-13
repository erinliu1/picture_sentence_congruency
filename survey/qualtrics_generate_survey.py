from __future__ import annotations

import copy
import html
import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

TEMPLATE_QSF_PATH = Path("qualtrics_templates.qsf")
IMAGE_MAPPING_PATH = Path("qualtrics_image_mapping.json")
ASSIGNMENTS_DIRECTORY = Path("survey_assignments")
OUTPUT_DIRECTORY = Path("generated_qsf")

SURVEY_NAMES = ("A", "B", "C", "D")

# The manually formatted question in the template to duplicate.
QUESTION_TEMPLATE_ID = "QID8"

# The block in which generated questions should be placed.
EXPERIMENT_BLOCK_NAME = "Experiment"

# Insert a page break after every N experimental questions.
QUESTIONS_PER_PAGE = 10

# Generated question IDs begin here to avoid collisions with template IDs.
FIRST_GENERATED_QID_NUMBER = 1001


# ============================================================
# GENERAL HELPERS
# ============================================================

def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_path.replace(path)


def normalize_filename(value: str) -> str:
    return Path(value.strip()).name.lower()


def ensure_period(sentence: str) -> str:
    sentence = sentence.strip()

    if not sentence:
        raise ValueError("Encountered an empty sentence.")

    if sentence[-1] not in ".!?":
        sentence += "."

    return sentence


# ============================================================
# IMAGE MAPPING
# ============================================================

def extract_graphic_id(value: Any) -> str | None:
    """Find an IM_... graphic ID in a mapping value."""
    if isinstance(value, str):
        match = re.search(r"\bIM_[A-Za-z0-9]+\b", value)
        return match.group(0) if match else None

    if isinstance(value, dict):
        for key in (
            "graphic_id",
            "graphicId",
            "graphicID",
            "id",
            "image_id",
            "imageId",
            "url",
            "URL",
        ):
            if key in value:
                graphic_id = extract_graphic_id(value[key])

                if graphic_id:
                    return graphic_id

        for nested_value in value.values():
            graphic_id = extract_graphic_id(nested_value)

            if graphic_id:
                return graphic_id

    return None


def extract_image_url(value: Any) -> str | None:
    """Find a complete Qualtrics image URL in a mapping value."""
    if isinstance(value, str):
        if value.startswith(("https://", "http://")) and "IM_" in value:
            return value

        return None

    if isinstance(value, dict):
        for key in ("url", "URL", "image_url", "imageUrl"):
            candidate = value.get(key)

            if (
                isinstance(candidate, str)
                and candidate.startswith(("https://", "http://"))
                and "IM_" in candidate
            ):
                return candidate

        for nested_value in value.values():
            url = extract_image_url(nested_value)

            if url:
                return url

    return None


def flatten_mapping_object(
    obj: Any,
    output: dict[str, Any],
) -> None:
    """
    Recursively collect filename-like keys from several possible mapping formats.

    Supported examples include:

    {
      "library.png": {
        "graphic_id": "IM_..."
      }
    }

    {
      "images": {
        "library.png": {
          "id": "IM_..."
        }
      }
    }

    [
      {
        "filename": "library.png",
        "graphic_id": "IM_..."
      }
    ]
    """
    if isinstance(obj, dict):
        filename = None

        for key in (
            "filename",
            "file_name",
            "name",
            "image_filename",
            "local_filename",
        ):
            candidate = obj.get(key)

            if isinstance(candidate, str) and candidate:
                filename = candidate
                break

        if filename and extract_graphic_id(obj):
            output[normalize_filename(filename)] = obj

        for key, value in obj.items():
            if (
                isinstance(key, str)
                and Path(key).suffix.lower()
                in {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                and extract_graphic_id(value)
            ):
                output[normalize_filename(key)] = value

            flatten_mapping_object(value, output)

    elif isinstance(obj, list):
        for item in obj:
            flatten_mapping_object(item, output)


def load_image_mapping(path: Path) -> dict[str, Any]:
    raw_mapping = load_json(path)
    mapping: dict[str, Any] = {}

    flatten_mapping_object(raw_mapping, mapping)

    if not mapping:
        raise ValueError(
            f"No filename-to-graphic-ID entries could be found in {path}."
        )

    return mapping


def find_template_graphic_base_url(question_text: str) -> str:
    """
    Extract the template's Graphic.php URL prefix.

    Example result:
    https://mit.co1.qualtrics.com/CP/Graphic.php?IM=
    """
    match = re.search(
        r"""https?://[^"'<>]+?/CP/Graphic\.php\?IM=""",
        question_text,
    )

    if not match:
        raise ValueError(
            "Could not locate a Qualtrics Graphic.php URL in the template "
            "question."
        )

    return match.group(0)


def resolve_image_url(
    image_filename: str,
    mapping: dict[str, Any],
    graphic_base_url: str,
) -> str:
    normalized = normalize_filename(image_filename)

    value = mapping.get(normalized)

    if value is None:
        # Permit a mapping keyed by the filename stem instead of the full name.
        target_stem = Path(normalized).stem

        matches = [
            mapping_value
            for mapping_name, mapping_value in mapping.items()
            if Path(mapping_name).stem == target_stem
        ]

        if len(matches) == 1:
            value = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"Multiple image mapping entries match {image_filename!r}."
            )
        else:
            raise KeyError(
                f"No Qualtrics image mapping was found for "
                f"{image_filename!r}."
            )

    mapped_url = extract_image_url(value)

    if mapped_url:
        return mapped_url

    graphic_id = extract_graphic_id(value)

    if not graphic_id:
        raise ValueError(
            f"The mapping entry for {image_filename!r} does not contain "
            f"an IM_... graphic ID."
        )

    return f"{graphic_base_url}{graphic_id}"


# ============================================================
# QSF HELPERS
# ============================================================

def find_question_element(
    qsf: dict[str, Any],
    question_id: str,
) -> dict[str, Any]:
    for element in qsf.get("SurveyElements", []):
        if (
            element.get("Element") == "SQ"
            and element.get("PrimaryAttribute") == question_id
        ):
            return element

    raise KeyError(f"Could not find template question {question_id!r}.")


def find_experiment_block(
    qsf: dict[str, Any],
) -> dict[str, Any]:
    for element in qsf.get("SurveyElements", []):
        if element.get("Element") != "BL":
            continue

        payload = element.get("Payload", [])

        if not isinstance(payload, list):
            continue

        for block in payload:
            if block.get("Description") == EXPERIMENT_BLOCK_NAME:
                return block

    raise KeyError(
        f"Could not find a block named {EXPERIMENT_BLOCK_NAME!r}."
    )


def build_question_text(
    *,
    image_url: str,
    sentence_frame: str,
    final_word: str,
) -> str:
    """
    Build the formatted image-and-sentence prompt.

    This reproduces the formatting of the manually created QID8 template:
    Georgia, 16 px, image, Sentence heading, blockquote, and bold final word.
    """
    sentence_frame = sentence_frame.rstrip()
    final_word = final_word.strip()

    punctuation = ""

    if final_word and final_word[-1] in ".!?":
        punctuation = final_word[-1]
        final_word = final_word[:-1]
    else:
        punctuation = "."

    escaped_frame = html.escape(sentence_frame)
    escaped_word = html.escape(final_word)
    escaped_url = html.escape(image_url, quote=True)

    return (
        '<span style="font-size:16px;">'
        '<span style="font-family:Georgia,serif;">'
        f'<img src="{escaped_url}" '
        'style="width: 512px; height: 512px;" />'
        "</span></span>\n"
        '<p><span style="font-size:16px;">'
        '<span style="font-family:Georgia,serif;">'
        "<strong>Sentence</strong>"
        "</span></span></p>\n\n"
        "<blockquote>\n"
        '<p><span style="font-size:16px;">'
        '<span style="font-family:Georgia,serif;">'
        f"{escaped_frame} <strong>{escaped_word}</strong>"
        f"{html.escape(punctuation)}"
        "</span></span></p>\n"
        "</blockquote>"
    )


def make_generated_question(
    *,
    template_question: dict[str, Any],
    question_id: str,
    export_tag: str,
    question_number: int,
    sample: dict[str, Any],
    image_url: str,
) -> dict[str, Any]:
    question = copy.deepcopy(template_question)

    sentence_frame = sample["sentence_frame"]
    final_word = sample["final_word"]
    sentence = ensure_period(
        sample.get("sentence")
        or f"{sentence_frame.rstrip()} {final_word.strip()}"
    )

    description = f"Sentence {sentence}"

    question["PrimaryAttribute"] = question_id
    question["SecondaryAttribute"] = description

    payload = question["Payload"]
    payload["QuestionID"] = question_id
    payload["DataExportTag"] = export_tag
    payload["QuestionDescription"] = description
    payload["QuestionText"] = build_question_text(
        image_url=image_url,
        sentence_frame=sentence_frame,
        final_word=final_word,
    )

    # Preserve all answer options and formatting from the template.
    return question


def update_question_count(qsf: dict[str, Any]) -> None:
    number_of_questions = sum(
        element.get("Element") == "SQ"
        for element in qsf.get("SurveyElements", [])
    )

    for element in qsf.get("SurveyElements", []):
        if (
            element.get("Element") == "QC"
            or element.get("PrimaryAttribute") == "Survey Question Count"
        ):
            element["SecondaryAttribute"] = str(number_of_questions)


def update_survey_name(
    qsf: dict[str, Any],
    survey_name: str,
) -> None:
    entry = qsf.get("SurveyEntry", {})
    original_name = entry.get(
        "SurveyName",
        "Picture-Sentence Compatibility Ratings",
    )

    # Avoid accumulating suffixes if the source template was itself generated.
    original_name = re.sub(
        r"\s*[-–—]\s*Survey\s+[A-D]\s*$",
        "",
        original_name,
        flags=re.IGNORECASE,
    )

    entry["SurveyName"] = f"{original_name} – Survey {survey_name}"


# ============================================================
# ASSIGNMENT HELPERS
# ============================================================

def load_survey_questions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)

    if isinstance(data, list):
        questions = data
    elif isinstance(data, dict):
        if isinstance(data.get("questions"), list):
            questions = data["questions"]
        elif isinstance(data.get("items"), list):
            questions = data["items"]
        else:
            raise ValueError(
                f"{path} must contain a 'questions' list."
            )
    else:
        raise ValueError(f"Unexpected JSON structure in {path}.")

    required_fields = {
        "item_index",
        "sentence_frame",
        "final_word",
        "image_filename",
        "congruency",
    }

    for index, question in enumerate(questions, start=1):
        missing = required_fields - question.keys()

        if missing:
            raise ValueError(
                f"Question {index} in {path} is missing fields: "
                f"{sorted(missing)}"
            )

    question_numbers = [
        question.get("question_number")
        for question in questions
    ]

    if all(number is not None for number in question_numbers):
        questions = sorted(
            questions,
            key=lambda question: question["question_number"],
        )

    return questions


# ============================================================
# SURVEY GENERATION
# ============================================================

def generate_one_survey(
    *,
    template_qsf: dict[str, Any],
    image_mapping: dict[str, Any],
    survey_name: str,
) -> dict[str, Any]:
    assignment_path = (
        ASSIGNMENTS_DIRECTORY / f"survey_{survey_name}.json"
    )
    samples = load_survey_questions(assignment_path)

    qsf = copy.deepcopy(template_qsf)
    template_question = find_question_element(
        qsf,
        QUESTION_TEMPLATE_ID,
    )
    experiment_block = find_experiment_block(qsf)

    template_question_text = template_question["Payload"]["QuestionText"]
    graphic_base_url = find_template_graphic_base_url(
        template_question_text
    )

    # Start the Experiment block from a clean slate.
    experiment_block["BlockElements"] = []

    existing_question_ids = {
        element.get("PrimaryAttribute")
        for element in qsf.get("SurveyElements", [])
        if element.get("Element") == "SQ"
    }

    next_qid_number = FIRST_GENERATED_QID_NUMBER
    generated_questions: list[dict[str, Any]] = []

    for question_number, sample in enumerate(samples, start=1):
        while f"QID{next_qid_number}" in existing_question_ids:
            next_qid_number += 1

        question_id = f"QID{next_qid_number}"
        export_tag = f"S{survey_name}_Q{question_number}"

        image_url = resolve_image_url(
            sample["image_filename"],
            image_mapping,
            graphic_base_url,
        )

        generated_question = make_generated_question(
            template_question=template_question,
            question_id=question_id,
            export_tag=export_tag,
            question_number=question_number,
            sample=sample,
            image_url=image_url,
        )

        generated_questions.append(generated_question)
        existing_question_ids.add(question_id)

        experiment_block["BlockElements"].append(
            {
                "Type": "Question",
                "QuestionID": question_id,
            }
        )

        if (
            question_number % QUESTIONS_PER_PAGE == 0
            and question_number != len(samples)
        ):
            experiment_block["BlockElements"].append(
                {"Type": "Page Break"}
            )

        next_qid_number += 1

    # Add generated SQ elements before the final STAT element when possible.
    survey_elements = qsf["SurveyElements"]
    insertion_index = next(
        (
            index
            for index, element in enumerate(survey_elements)
            if element.get("Element") == "STAT"
        ),
        len(survey_elements),
    )

    survey_elements[insertion_index:insertion_index] = generated_questions

    update_survey_name(qsf, survey_name)
    update_question_count(qsf)

    return qsf


def verify_generated_qsf(
    qsf: dict[str, Any],
    expected_experiment_questions: int,
) -> None:
    experiment_block = find_experiment_block(qsf)

    block_question_ids = [
        element["QuestionID"]
        for element in experiment_block.get("BlockElements", [])
        if element.get("Type") == "Question"
    ]

    if len(block_question_ids) != expected_experiment_questions:
        raise ValueError(
            f"Experiment block contains {len(block_question_ids)} "
            f"questions; expected {expected_experiment_questions}."
        )

    if len(block_question_ids) != len(set(block_question_ids)):
        raise ValueError(
            "The Experiment block contains duplicate question IDs."
        )

    survey_question_ids = {
        element.get("PrimaryAttribute")
        for element in qsf.get("SurveyElements", [])
        if element.get("Element") == "SQ"
    }

    missing_ids = set(block_question_ids) - survey_question_ids

    if missing_ids:
        raise ValueError(
            f"The following Experiment block questions have no matching SQ "
            f"element: {sorted(missing_ids)}"
        )

    expected_page_breaks = (
        (expected_experiment_questions - 1) // QUESTIONS_PER_PAGE
    )
    actual_page_breaks = sum(
        element.get("Type") == "Page Break"
        for element in experiment_block.get("BlockElements", [])
    )

    if actual_page_breaks != expected_page_breaks:
        raise ValueError(
            f"Experiment block contains {actual_page_breaks} page breaks; "
            f"expected {expected_page_breaks}."
        )


def main() -> None:
    template_qsf = load_json(TEMPLATE_QSF_PATH)
    image_mapping = load_image_mapping(IMAGE_MAPPING_PATH)

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print(
        f"Loaded {len(image_mapping)} mapped Qualtrics graphics "
        f"from {IMAGE_MAPPING_PATH}."
    )

    for survey_name in SURVEY_NAMES:
        assignment_path = (
            ASSIGNMENTS_DIRECTORY / f"survey_{survey_name}.json"
        )
        samples = load_survey_questions(assignment_path)

        generated_qsf = generate_one_survey(
            template_qsf=template_qsf,
            image_mapping=image_mapping,
            survey_name=survey_name,
        )

        verify_generated_qsf(
            generated_qsf,
            expected_experiment_questions=len(samples),
        )

        output_path = (
            OUTPUT_DIRECTORY
            / f"Picture-Sentence_Compatibility_Survey_{survey_name}.qsf"
        )
        save_json(output_path, generated_qsf)

        congruent_count = sum(
            sample["congruency"] == "congruent"
            for sample in samples
        )
        incongruent_count = sum(
            sample["congruency"] == "incongruent"
            for sample in samples
        )

        print(
            f"Survey {survey_name}: "
            f"{len(samples)} questions | "
            f"{congruent_count} congruent | "
            f"{incongruent_count} incongruent"
        )
        print(f"  Saved: {output_path}")

    print("\nFinished generating all four QSF files.")


if __name__ == "__main__":
    main()

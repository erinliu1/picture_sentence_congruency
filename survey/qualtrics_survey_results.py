
from pathlib import Path

import json
import uuid
import pandas as pd
from glob import glob

SURVEY_DIR = Path("/Intern/Erin/picture_sentence_congruency/survey")
OUTPUT_PATH = SURVEY_DIR / "results" / "behavior_human.csv"

all_survey_results = []
for survey_ID in ['A', 'B', 'C', 'D']:
    survey_json_path = SURVEY_DIR / "survey_assignments" / f"survey_{survey_ID}.json"
    survey_response_path = glob(str(SURVEY_DIR / "survey_responses" / f"Survey {survey_ID}_*.csv"))[0]

    with open(survey_json_path, 'r') as f:
        survey_data = json.load(f)

    survey_questions = survey_data['questions']
    survey_response = pd.read_csv(survey_response_path)

    participant_ids = [str(uuid.uuid4()) for _ in range(len(survey_response))]

    # save per-participant ratings
    for participant_idx, (_, participant) in enumerate(survey_response.iterrows()):
        participant_id = participant_ids[participant_idx]
        for question in survey_questions:
            question_id = question['question_number']
            congruency = question['congruency']
            item_index = question['item_index']
            image_word = question['image_filename'].split('.')[0]
            sentence = question['sentence']
            rating = participant[f'S{survey_ID}_Q{question_id}']
            # Participants who didn't finish the survey have NaN (a float)
            # for unanswered questions instead of a response string.
            if isinstance(rating, str) and rating[0] in ['1','2','3','4','5']:
                rating = rating[0]
                all_survey_results.append({
                    'participant_id': participant_id,
                    'item_index': item_index,
                    'image_word': image_word,
                    'condition': congruency,
                    'rating': int(rating)
                })

df = pd.DataFrame(all_survey_results)
df = df.sort_values(by=['participant_id', 'item_index', 'condition', 'image_word']).reset_index(drop=True)
df.to_csv(OUTPUT_PATH, index=False)
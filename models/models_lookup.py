import json

# add more models here

models_lookup = {
    'qwen3_vl_2b_instruct': {
        "title": "Qwen3-VL-2B-Instruct",
        "color": "#A3E635",
    },
    'qwen3_vl_4b_instruct': {
        "title": "Qwen3-VL-4B-Instruct",
        "color": "#84CC16",
    },
    'qwen3_vl_8b_instruct': {
        "title": "Qwen3-VL-8B-Instruct",
        "color": "#65A30D",
    },
    'qwen3_vl_32b_instruct': {
        "title": "Qwen3-VL-32B-Instruct",
        "color": "#16A34A",
    },
    'qwen3_vl_30b_a3b_instruct': {
        "title": "Qwen3-VL-30B-A3B-Instruct",
        "color": "#047857",
    },
    # 'qwen3_vl_235b_a22b_instruct': {
    #     "title": "Qwen3-VL-235B-A22B-Instruct",
    #     "color": "#064E3B",
    # } <-- this guy doesnn't work 
}

with open("/Intern/Erin/picture_sentence_congruency/models/models_lookup.json", "w", encoding="utf-8") as file:
    json.dump(models_lookup, file, indent=4, ensure_ascii=False)
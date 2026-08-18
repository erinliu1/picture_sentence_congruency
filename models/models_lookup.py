import json

# add more models here

models_lookup = {
    'qwen3_vl_2b_instruct': {
        "title": "Qwen3-VL-2B-Instruct",
        "color": "#BEF264",
        "family": "qwen",
        "type": "dense",
        "size": "2B",
    },
    'qwen3_vl_4b_instruct': {
        "title": "Qwen3-VL-4B-Instruct",
        "color": "#84CC16",
        "family": "qwen",
        "type": "dense",
        "size": "4B",
    },
    'qwen3_vl_8b_instruct': {
        "title": "Qwen3-VL-8B-Instruct",
        "color": "#4D7C0F",
        "family": "qwen",
        "type": "dense",
        "size": "8B",
    },
    'qwen3_vl_32b_instruct': {
        "title": "Qwen3-VL-32B-Instruct",
        "color": "#15803D",
        "family": "qwen",
        "type": "dense",
        "size": "32B",
    },
    'qwen3_vl_30b_a3b_instruct': {
        "title": "Qwen3-VL-30B-A3B-Instruct",
        "color": "#14532D",
        "family": "qwen",
        "type": "moe",
        "size": "30B-A4B",
    },   
    # 'qwen3_vl_235b_a22b_instruct': {
    #     "title": "Qwen3-VL-235B-A22B-Instruct",
    #     "color": "#064E3B",
    #     "family": "qwen"
    # } <-- this guy doesn't work

    'internvl3_1b_instruct': {
        "title": "InternVL3-1B-Instruct",
        "color": "#BFDBFE",
        "family": "intern",
        "type": "dense",
        "size": "1B",
    },
    'internvl3_2b_instruct': {
        "title": "InternVL3-2B-Instruct",
        "color": "#93C5FD",
        "family": "intern",
        "type": "dense",
        "size": "2B",
    },
    'internvl3_8b_instruct': {
        "title": "InternVL3-8B-Instruct",
        "color": "#60A5FA",
        "family": "intern",
        "type": "dense",
        "size": "8B",
    },
    'internvl3_14b_instruct': {
        "title": "InternVL3-14B-Instruct",
        "color": "#3B82F6",
        "family": "intern",
        "type": "dense",
        "size": "14B",
    },
    'internvl3_38b_instruct': {
        "title": "InternVL3-38B-Instruct",
        "color": "#2563EB",
        "family": "intern",
        "type": "dense",
        "size": "38B",
    },
    'internvl3_78b_instruct': {
        "title": "InternVL3-78B-Instruct",
        "color": "#1E3A8A",
        "family": "intern",
        "type": "dense",
        "size": "78B",
    },

    'plm_1b': {
        "title": "PLM-1B",
        "color": "#DDD6FE",
        "family": "plm",
        "type": "dense",
        "size": "1B",
    },
    'plm_3b': {
        "title": "PLM-3B",
        "color": "#A78BFA",
        "family": "plm",
        "type": "dense",
        "size": "3B",
    },
    'plm_8b': {
        "title": "PLM-8B",
        "color": "#6D28D9",
        "family": "plm",
        "type": "dense",
        "size": "8B",
    },
    'llava_ov_05b': {
        "title": "LLaVA-OV-0.5B",
        "color": "#FBCFE8",
        "family": "llava",
        "type": "dense",
        "size": "0.5B",
        },
    'llava_ov_7b': {
        "title": "LLaVA-OV-7B",
        "color": "#F472B6",
        "family": "llava",
        "type": "dense",
        "size": "7B",
    },
    'llava_ov_72b': {
        "title": "LLaVA-OV-72B",
        "color": "#BE185D",
        "family": "llava",
        "type": "dense",
        "size": "72B",
    },
    'glm_9b': {
        "title": "GLM-4.6V-9B",
        "color": "#67E8F9",
        "family": "glm",
        "type": "dense",
        "size": "9B",
    },
    # 'glm_106b_a12b': {
    #     "title": "GLM-4.6V-106B-A12B",
    #     "color": "#0E7490",
    #     "family": "glm",
    #     "type": "moe",
    #     "size": "106-A12B",
    # }, <-- this guy doesn't work

    'gemma3_4b': {
        "title": "Gemma3-4B",
        "color": "#FDE68A",
        "family": "gemma3",
        "type": "dense",
        "size": "4B",
    },
    'gemma3_12b': {
        "title": "Gemma3-12B",
        "color": "#F59E0B",
        "family": "gemma3",
        "type": "dense",
        "size": "12B",
    },
    'gemma3_27b': {
        "title": "Gemma3-27B",
        "color": "#B45309",
        "family": "gemma3",
        "type": "dense",
        "size": "27B",
    },

    'gemma4_e2b': {
        "title": "Gemma4-E2B",
        "color": "#FFEDD5",
        "family": "gemma4",
        "type": "PLE",
        "size": "E2B",
    },
    'gemma4_e4b': {
        "title": "Gemma4-E4B",
        "color": "#FDBA74",
        "family": "gemma4",
        "type": "PLE",
        "size": "E4B",
    },
    'gemma4_12b': {
        "title": "Gemma4-12B",
        "color": "#FB923C",
        "family": "gemma4",
        "type": "dense",
        "size": "12B",
    },
    'gemma4_26b_a4b': {
        "title": "Gemma4-26B-A4B",
        "color": "#EA580C",
        "family": "gemma4",
        "type": "moe",
        "size": "26B-A4B",
    },
    'gemma4_31b': {
        "title": "Gemma4-31B",
        "color": "#7C2D12",
        "family": "gemma4",
        "type": "dense",
        "size": "31B",
    },
}

with open("/Intern/Erin/picture_sentence_congruency/models/models_lookup.json", "w", encoding="utf-8") as file:
    json.dump(models_lookup, file, indent=4, ensure_ascii=False)
import json

# add more models here

models_lookup = {
    'qwen3_vl_2b_instruct': {
        "title": "Qwen3-VL-2B-Instruct",
        "color": "#D9F99D",
        "family": "qwen",
        "type": "dense",
        "size": "2B",
    },
    'qwen3_vl_4b_instruct': {
        "title": "Qwen3-VL-4B-Instruct",
        "color": "#A3E635",
        "family": "qwen",
        "type": "dense",
        "size": "4B",
    },
    'qwen3_vl_8b_instruct': {
        "title": "Qwen3-VL-8B-Instruct",
        "color": "#65A30D",
        "family": "qwen",
        "type": "dense",
        "size": "8B",
    },
    'qwen3_vl_32b_instruct': {
        "title": "Qwen3-VL-32B-Instruct",
        "color": "#4D7C0F",
        "family": "qwen",
        "type": "dense",
        "size": "32B",
    },
    'qwen3_vl_30b_a3b_instruct': {
        "title": "Qwen3-VL-30B-A3B-Instruct",
        "color": "#3F6212",
        "family": "qwen",
        "type": "moe",
        "size": "30B-A3B",
    },
    'qwen3_vl_235b_a22b_instruct': {
        "title": "Qwen3-VL-235B-A22B-Instruct",
        "color": "#365314",
        "family": "qwen",
        "type": "moe",
        "size": "235B-A22B",
    },
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

    # Base (pretrained, non-instruction-tuned) counterparts of internvl3_{1b,2b,8b,14b,38b,78b}, the
    # OpenGVLab/InternVL3-{size}-Pretrained checkpoints. Unlike the "-hf" instruct checkpoints (which use
    # transformers-native InternVLForConditionalGeneration + AutoProcessor), these ship only in OpenGVLab's
    # original custom-code format (InternVLChatModel via trust_remote_code=True), so hidden-state extraction
    # for this family follows a different code path (see hs_intern.py). Kept in a separate "intern_base"
    # family for the same reason as gemma3_base/gemma4_base.
    'internvl3_1b_base': {
        "title": "InternVL3-1B-Base",
        "color": "#E5E7EB",
        "family": "intern_base",
        "type": "dense",
        "size": "1B",
    },
    'internvl3_2b_base': {
        "title": "InternVL3-2B-Base",
        "color": "#D1D5DB",
        "family": "intern_base",
        "type": "dense",
        "size": "2B",
    },
    'internvl3_8b_base': {
        "title": "InternVL3-8B-Base",
        "color": "#9CA3AF",
        "family": "intern_base",
        "type": "dense",
        "size": "8B",
    },
    'internvl3_14b_base': {
        "title": "InternVL3-14B-Base",
        "color": "#6B7280",
        "family": "intern_base",
        "type": "dense",
        "size": "14B",
    },
    'internvl3_38b_base': {
        "title": "InternVL3-38B-Base",
        "color": "#4B5563",
        "family": "intern_base",
        "type": "dense",
        "size": "38B",
    },
    'internvl3_78b_base': {
        "title": "InternVL3-78B-Base",
        "color": "#1F2937",
        "family": "intern_base",
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
    'glm_106b_a12b': {
        "title": "GLM-4.6V-106B-A12B",
        "color": "#0E7490",
        "family": "glm",
        "type": "moe",
        "size": "106B-A12B",
    },
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

    # Base (pretrained, non-instruction-tuned) counterparts of gemma3_{4b,12b,27b}, the
    # google/gemma-3-{4b,12b,27b}-pt checkpoints. Kept in a separate "gemma3_base" family
    # (rather than folded into "gemma3") so existing gemma3-family aggregate plots/scripts
    # aren't silently changed by an untuned model.
    'gemma3_4b_base': {
        "title": "Gemma3-4B-Base",
        "color": "#9CA3AF",
        "family": "gemma3_base",
        "type": "dense",
        "size": "4B",
    },
    'gemma3_12b_base': {
        "title": "Gemma3-12B-Base",
        "color": "#6B7280",
        "family": "gemma3_base",
        "type": "dense",
        "size": "12B",
    },
    'gemma3_27b_base': {
        "title": "Gemma3-27B-Base",
        "color": "#374151",
        "family": "gemma3_base",
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

    # Base (pretrained, non-instruction-tuned) counterparts of gemma4_{e2b,e4b,12b,26b_a4b,31b},
    # the google/gemma-4-{E2B,E4B,12B,26B-A4B,31B} checkpoints (no "-it" suffix). Kept in a
    # separate "gemma4_base" family for the same reason as gemma3_base: so existing gemma4-family
    # aggregate plots/scripts aren't silently changed by an untuned model.
    'gemma4_e2b_base': {
        "title": "Gemma4-E2B-Base",
        "color": "#D1D5DB",
        "family": "gemma4_base",
        "type": "PLE",
        "size": "E2B",
    },
    'gemma4_e4b_base': {
        "title": "Gemma4-E4B-Base",
        "color": "#9CA3AF",
        "family": "gemma4_base",
        "type": "PLE",
        "size": "E4B",
    },
    'gemma4_12b_base': {
        "title": "Gemma4-12B-Base",
        "color": "#6B7280",
        "family": "gemma4_base",
        "type": "dense",
        "size": "12B",
    },
    'gemma4_26b_a4b_base': {
        "title": "Gemma4-26B-A4B-Base",
        "color": "#4B5563",
        "family": "gemma4_base",
        "type": "moe",
        "size": "26B-A4B",
    },
    'gemma4_31b_base': {
        "title": "Gemma4-31B-Base",
        "color": "#1F2937",
        "family": "gemma4_base",
        "type": "dense",
        "size": "31B",
    },
}

with open("/Intern/Erin/picture_sentence_congruency/models/models_lookup.json", "w", encoding="utf-8") as file:
    json.dump(models_lookup, file, indent=4, ensure_ascii=False)
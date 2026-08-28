# For every stimulus, feed to each InternVL3 model, pull out the model's internal hidden states at the last word of the sentence, for every layer.
# Saves a .pt file per stimulus, naming convention as {item_index}_{image_word}_{condition}.pt
# item_index is the order of the stimulus in the sentences.json

from __future__ import annotations

import glob
import json
import os
import random

import numpy as np
import torch

from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from prepare_cuda import prepare_cuda
prepare_cuda(allow_multi_gpu=False)

import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer, InternVLForConditionalGeneration

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

STIMULI_PATH = "/Intern/Erin/picture_sentence_congruency/stimuli/sentences.json"
PICTURES_DIR = Path("/Intern/Erin/picture_sentence_congruency/stimuli/pictures")

with open(STIMULI_PATH, "r") as f:
    sentences = json.load(f)

MODELS_DIR = Path("/Intern/Erin/picture_sentence_congruency/models")
with open(MODELS_DIR / "models_lookup.json", "r", encoding="utf-8") as file:
    models_lookup = json.load(file)

# All InternVL3 checkpoints are dense, run on a single GPU, and tokenize rating digits "1"-"5" as single tokens.
INTERNVL_DENSE_LOOKUP = {
    "internvl3_1b_instruct": "OpenGVLab/InternVL3-1B-hf",
    "internvl3_2b_instruct": "OpenGVLab/InternVL3-2B-hf",
    "internvl3_8b_instruct": "OpenGVLab/InternVL3-8B-hf",
    "internvl3_14b_instruct": "OpenGVLab/InternVL3-14B-hf",
    "internvl3_38b_instruct": "OpenGVLab/InternVL3-38B-hf",
    "internvl3_78b_instruct": "OpenGVLab/InternVL3-78B-hf",
}

# Base (pretrained, non-instruction-tuned) counterparts -- OpenGVLab/InternVL3-{size}-Pretrained.
# Unlike every other base family in this project, these do NOT ship a transformers-native "-hf"
# checkpoint -- only OpenGVLab's original custom-code implementation (InternVLChatModel, loaded via
# trust_remote_code=True) exists for the pretrained stage. That means a different model class,
# tokenizer, and image-preprocessing path (manual dynamic tiling, per the model cards) rather than
# the AutoProcessor used for the "-hf" instruct checkpoints above.
INTERNVL_BASE_LOOKUP = {
    "internvl3_1b_base": "OpenGVLab/InternVL3-1B-Pretrained",
    "internvl3_2b_base": "OpenGVLab/InternVL3-2B-Pretrained",
    "internvl3_8b_base": "OpenGVLab/InternVL3-8B-Pretrained",
    "internvl3_14b_base": "OpenGVLab/InternVL3-14B-Pretrained",
    "internvl3_38b_base": "OpenGVLab/InternVL3-38B-Pretrained",
    "internvl3_78b_base": "OpenGVLab/InternVL3-78B-Pretrained",
}

# --- Dynamic-tiling image preprocessing for the base (trust_remote_code) checkpoints, copied from
# the standard InternVL model-card usage snippet (same code across all OpenGVLab InternVL repos). ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN = "<img>", "</img>", "<IMG_CONTEXT>"

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image_tiles(image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(tile) for tile in tiles])
    return pixel_values
# --- end dynamic-tiling preprocessing ---

def patch_internvl_post_init(model_name):
    # OpenGVLab's InternVLChatModel.__init__ (the trust_remote_code class backing every
    # "-Pretrained" checkpoint) never calls self.post_init(), a call current transformers
    # (5.15.0 here) unconditionally relies on to populate self.all_tied_weights_keys during
    # from_pretrained's loading finalization -- without it, loading crashes with
    # "'InternVLChatModel' object has no attribute 'all_tied_weights_keys'". Pre-download the repo
    # and patch that one line into both cached copies of modeling_internvl_chat.py (the raw hub
    # snapshot and the transformers_modules dynamic-import copy) before AutoModel ever imports the
    # class. Idempotent -- skips files that are already patched.
    from huggingface_hub import snapshot_download
    snapshot_download(model_name)

    candidates = glob.glob(
        f"{os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))}/**/modeling_internvl_chat.py",
        recursive=True,
    )
    repo_hint = model_name.split("/")[-1]
    marker = "self.system_message = self.conv_template.system_message\n"
    for path in candidates:
        if repo_hint.replace("-", "_") not in path and repo_hint not in path:
            continue
        with open(path, "r") as f:
            content = f.read()
        if "self.post_init()" in content:
            continue
        if marker not in content:
            continue
        content = content.replace(marker, marker + "        self.post_init()\n", 1)
        with open(path, "w") as f:
            f.write(content)

def extract_hidden_states(model_id):
    if model_id in INTERNVL_DENSE_LOOKUP:
        model_class = InternVLForConditionalGeneration
        model_name = INTERNVL_DENSE_LOOKUP[model_id]
        is_base = False
        dtype = torch.bfloat16
        device_map = {"": 0}
    elif model_id in INTERNVL_BASE_LOOKUP:
        model_name = INTERNVL_BASE_LOOKUP[model_id]
        is_base = True
        dtype = torch.bfloat16
        device_map = {"": 0}
    else:
        return
    title = models_lookup[model_id]["title"]

    HIDDEN_STATES_DIR = Path(f"/Intern/Erin/picture_sentence_congruency/models/{model_id}/hidden_states")
    HIDDEN_STATES_DIR.mkdir(parents=True, exist_ok=True)

    if is_base:
        patch_internvl_post_init(model_name)
        model = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        period_ids = tokenizer.encode(".", add_special_tokens=False)
        if len(period_ids) != 1:
            raise ValueError(f"expected a single token ID for the period, but {model_id} gave {period_ids}")
        PERIOD_TOKEN_ID = period_ids[0]
    else:
        model = model_class.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            attn_implementation="sdpa"
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(model_name)

        period_ids = processor.tokenizer.encode(".", add_special_tokens=False)
        if len(period_ids) != 1:
            raise ValueError(f"expected a single token ID for the period, but {model_id} gave {period_ids}")
        PERIOD_TOKEN_ID = period_ids[0]

    for item_index, item in enumerate(tqdm(sentences, desc=f"Extracting hidden states for {title}")):
        sentence_frame = item['sentence_frame']
        word_a, word_b = item['word_options']

        congruent_pairs = [(word_a, word_a), (word_b, word_b)]
        incongruent_pairs = [(word_a, word_b), (word_b, word_a)]
        for image_word, final_word in congruent_pairs + incongruent_pairs:
            is_congruent = (image_word == final_word)
            condition = 'congruent' if is_congruent else 'incongruent'

            sentence = f"{sentence_frame} {final_word}."
            image_path = PICTURES_DIR / f"{image_word}.png"
            image = Image.open(image_path).convert("RGB")

            if is_base:
                pixel_values = load_image_tiles(image, max_num=12).to(device=model.device, dtype=dtype)
                num_patches = pixel_values.shape[0]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
                query = f"<image>\n{sentence}".replace("<image>", image_tokens, 1)

                model_inputs = tokenizer(query, return_tensors="pt")
                input_ids = model_inputs["input_ids"].to(model.device)
                attention_mask = model_inputs["attention_mask"].to(model.device)
                image_flags = torch.ones((num_patches, 1), dtype=torch.long, device=model.device)

                token_ids = input_ids[0]
                period_index = (token_ids == PERIOD_TOKEN_ID).nonzero(as_tuple=True)[0][-1].item()
                critical_word_final_token_index = period_index - 1

                with torch.inference_mode():
                    outputs = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        image_flags=image_flags,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {
                                "type": "text",
                                "text": sentence,
                            },
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_tensors="pt",
                )

                token_ids = inputs["input_ids"][0]
                period_index = (token_ids == PERIOD_TOKEN_ID).nonzero(as_tuple=True)[0][-1].item()
                critical_word_final_token_index = period_index - 1

                inputs = inputs.to(model.device)
                with torch.inference_mode():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        use_cache=False,
                    )

            hidden_states = outputs.hidden_states[1:] # exclude embedding state 0

            critical_word_hidden_states = torch.stack([layer_hidden_state[0, critical_word_final_token_index] for layer_hidden_state in hidden_states], dim=0).to(device="cpu", dtype=torch.float32)  # num_layers x hidden_size

            name = f"{item_index}_{image_word}_{condition}.pt"
            path = HIDDEN_STATES_DIR / name
            torch.save(critical_word_hidden_states, path)

    print(f"Hidden states for {model_id} saved.")
    del model
    torch.cuda.empty_cache()
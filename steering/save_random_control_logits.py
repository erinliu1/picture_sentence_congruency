"""Generate, apply, and summarize norm-matched random orthogonal controls.

For each cross-validation fold at the peak RR-decoding layer, this script:

1. converts the fitted standardized-space probe direction into raw hidden-state
   coordinates;
2. samples reproducible random unit directions orthogonal to that raw direction;
3. norm-matches every random step to one true ``s_train`` steering step;
4. applies ``k`` multiples of each random step to the final prompt token; and
5. saves rating logits and a stimulus-set bootstrap summary.

Fold artifacts live under
``models/<model_id>/steering/layer<layer>/fold<fold>``. The aggregate summary
lives under ``steering/results/<model_id>``.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_vl_8b_instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
K_VALUES = [-2, -1, 0, 1, 2]
FOLDS = range(1, 11)
N_RANDOM_DIRECTIONS = 20
DIRECTION_SEED = 20261009
N_BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20261010
GPU_INDEX = os.environ.get("STEERING_GPU", "1")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_INDEX
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HF_HOME", str(ROOT_DIR.parent / "models/huggingface"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT_DIR.parent / "tmp/triton"))
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(ROOT_DIR.parent / "tmp/torchinductor"))
sys.path.insert(0, str(ROOT_DIR / "hidden_states"))
from prepare_cuda import prepare_cuda

prepare_cuda(allow_multi_gpu=True)
sys.path.insert(0, str(ROOT_DIR / "behavioral"))

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_prompt import build_rating_inputs


def get_peak_layer():
    """Return the earliest layer attaining peak all-condition RR accuracy."""
    path = ROOT_DIR / "models" / MODEL_ID / "decoding_instruction/layerwise_accuracy.csv"
    accuracy = pd.read_csv(path).sort_values("layer_index").reset_index(drop=True)
    if accuracy.empty or accuracy[["layer_index", "all_accuracy"]].isna().any().any():
        raise ValueError(f"Incomplete decoding accuracy table: {path}")
    return int(accuracy.loc[accuracy.all_accuracy.idxmax(), "layer_index"])


def get_prompt(processor, stimuli, item_index, image_word, condition):
    """Build the exact behavioral instruction prompt for one held-out item."""
    stimulus = stimuli[int(item_index)]
    if condition not in {"congruent", "incongruent"}:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "congruent":
        final_word = image_word
    else:
        final_word = next(word for word in stimulus["word_options"] if word != image_word)
    sentence = f"{stimulus['sentence_frame']} {final_word}."
    image_path = ROOT_DIR / "stimuli/pictures" / f"{image_word}.png"
    with Image.open(image_path) as image:
        return build_rating_inputs(processor, image.convert("RGB"), sentence)


def get_rating_token_ids(processor):
    """Return the vocabulary IDs of the five single-token rating responses."""
    rating_ids = []
    for rating in range(1, 6):
        ids = processor.tokenizer.encode(str(rating), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Rating {rating} is not one token: {ids}")
        rating_ids.append(ids[0])
    return rating_ids


def summarize_logits(logits, rating_token_ids):
    """Return rating logits, valid-digit mass, and high-versus-low contrast."""
    rating_logits = logits[rating_token_ids].float()
    values = rating_logits.cpu().tolist()
    row = {f"l{rating}": value for rating, value in enumerate(values, 1)}
    row["valid_digit_mass"] = logits.float().softmax(dim=-1)[rating_token_ids].sum().item()
    row["logit_contrast"] = (row["l4"] + row["l5"] - row["l1"] - row["l2"]) / 2
    return row


def repeat_inputs(inputs, count):
    """Repeat one unpadded multimodal prompt into a batch of identical prompts."""
    return {
        key: value.repeat((count,) + (1,) * (value.ndim - 1))
        for key, value in inputs.items()
    }


def make_random_orthogonal_directions(fold, layer):
    """Create and save raw-space orthogonal, norm-matched control directions."""
    fold_dir = (
        ROOT_DIR / "models" / MODEL_ID / "steering" / f"layer{layer}" / f"fold{fold}"
    )
    pipeline = joblib.load(fold_dir / "probe.joblib")
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    np.testing.assert_array_equal(classifier.classes_, [0, 1])
    w = classifier.coef_[0]
    w_norm = np.linalg.norm(w)
    if not np.isfinite(w_norm) or w_norm == 0:
        raise ValueError(f"Invalid probe direction for fold {fold}, layer {layer}")
    w_hat = w / w_norm

    congruent = np.load(fold_dir / "congruent_projections.npy", allow_pickle=False)
    incongruent = np.load(fold_dir / "incongruent_projections.npy", allow_pickle=False)
    projections = np.concatenate([congruent, incongruent]).astype(np.float64)
    s_train = projections.std(ddof=1)
    if not np.isfinite(s_train) or s_train <= 0:
        raise ValueError(f"Invalid s_train for fold {fold}, layer {layer}")

    raw_direction = scaler.scale_ * w_hat
    true_unit_direction = raw_direction / np.linalg.norm(raw_direction)
    true_unit_step = s_train * raw_direction
    rng = np.random.default_rng(np.random.SeedSequence([DIRECTION_SEED, fold]))
    random_unit_directions = rng.normal(
        size=(N_RANDOM_DIRECTIONS, len(raw_direction))
    )
    random_unit_directions -= (
        (random_unit_directions @ true_unit_direction)[:, None]
        * true_unit_direction[None, :]
    )
    random_unit_directions /= np.linalg.norm(random_unit_directions, axis=1)[:, None]
    np.testing.assert_allclose(
        random_unit_directions @ true_unit_direction, 0, atol=1e-12
    )
    random_unit_steps = random_unit_directions * np.linalg.norm(true_unit_step)
    np.testing.assert_allclose(
        np.linalg.norm(random_unit_steps, axis=1),
        np.linalg.norm(true_unit_step),
        rtol=1e-12,
    )

    output_path = fold_dir / "random_orthogonal_directions.npz"
    np.savez(
        output_path,
        random_indices=np.arange(1, N_RANDOM_DIRECTIONS + 1),
        random_unit_directions=random_unit_directions,
        random_unit_steps=random_unit_steps,
        true_unit_direction=true_unit_direction,
        true_unit_step=true_unit_step,
        s_train=s_train,
        seed=DIRECTION_SEED,
        fold=fold,
        layer=layer,
    )
    return random_unit_steps


def get_random_control_logits(
    model, inputs, layer, k, random_unit_steps, rating_token_ids
):
    """Apply one k level of all random controls and return their rating metrics."""
    if k == 0:
        raise ValueError("Use saved baseline logits for k=0")
    deltas = torch.as_tensor(
        k * random_unit_steps, device=model.device, dtype=torch.float32
    )
    batched_inputs = repeat_inputs(inputs.to(model.device), N_RANDOM_DIRECTIONS)
    hook_calls = []

    def steer_final_token(module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        hook_calls.append(1)
        steered = hidden.clone()
        steered[:, -1, :] = (hidden[:, -1, :].float() + deltas).to(hidden.dtype)
        if isinstance(output, tuple):
            return (steered, *output[1:])
        if isinstance(output, list):
            return [steered, *output[1:]]
        return steered

    handle = model.model.language_model.layers[layer].register_forward_hook(
        steer_final_token
    )
    try:
        with torch.inference_mode():
            outputs = model(
                **batched_inputs, return_dict=True, use_cache=False, logits_to_keep=1
            )
    finally:
        handle.remove()
    if len(hook_calls) != 1:
        raise AssertionError(f"Expected one hook call, observed {len(hook_calls)}")
    return [
        summarize_logits(logits, rating_token_ids)
        for logits in outputs.logits[:, -1, :]
    ]


def get_held_out_names(fold):
    """Return one fold's held-out stimulus filenames in canonical CV order."""
    with (ROOT_DIR / "decoding/cv/cv_lookup_pt.json").open(encoding="utf-8") as file:
        split = json.load(file)[str(fold)]
    if set(split["train"]) & set(split["test"]):
        raise ValueError(f"Train/test stimulus sets overlap in fold {fold}")
    return [name for names in split["test"].values() for name in names]


def process_fold(model, processor, stimuli, rating_token_ids, fold, layer):
    """Generate directions and save random-control logits for one fold."""
    fold_dir = (
        ROOT_DIR / "models" / MODEL_ID / "steering" / f"layer{layer}" / f"fold{fold}"
    )
    baseline = pd.read_csv(fold_dir / "baseline_logits.csv")
    baseline = baseline.set_index(["item_index", "image_word", "condition"])
    if not baseline.index.is_unique:
        raise ValueError(f"Duplicate baseline rows in {fold_dir / 'baseline_logits.csv'}")
    random_unit_steps = make_random_orthogonal_directions(fold, layer)
    rows = []
    for name in get_held_out_names(fold):
        item_index, image_word, condition = Path(name).stem.split("_")
        metadata = {
            "item_index": int(item_index),
            "image_word": image_word,
            "condition": condition,
        }
        baseline_row = baseline.loc[(int(item_index), image_word, condition)]
        inputs = get_prompt(processor, stimuli, int(item_index), image_word, condition)
        for k in K_VALUES:
            if k == 0:
                metrics = [
                    baseline_row[
                        ["valid_digit_mass", "l1", "l2", "l3", "l4", "l5", "logit_contrast"]
                    ].to_dict()
                    for _ in range(N_RANDOM_DIRECTIONS)
                ]
            else:
                metrics = get_random_control_logits(
                    model, inputs, layer, k, random_unit_steps, rating_token_ids
                )
            for random_index, values in enumerate(metrics, 1):
                rows.append({
                    "random_index": random_index,
                    **metadata,
                    "k": k,
                    **values,
                })

    columns = [
        "random_index", "item_index", "image_word", "condition", "k",
        "valid_digit_mass", "l1", "l2", "l3", "l4", "l5", "logit_contrast",
    ]
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(fold_dir / "random_control_logits.csv", index=False)
    return table, baseline.reset_index()


def summarize_random_controls(control_tables, baseline_tables, layer):
    """Save mean baseline-relative effects and stimulus-set bootstrap intervals."""
    controls = pd.concat(control_tables, ignore_index=True)
    baselines = pd.concat(baseline_tables, ignore_index=True)
    keys = ["fold", "item_index", "image_word", "condition"]
    controls = controls.merge(
        baselines[keys + ["logit_contrast"]],
        on=keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_baseline"),
    )
    controls["delta_logit_contrast"] = (
        controls.logit_contrast - controls.logit_contrast_baseline
    )
    item_indices = np.sort(controls.item_index.unique())
    if len(item_indices) != 80:
        raise ValueError("Expected 80 unique held-out stimulus sets")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(item_indices), (N_BOOTSTRAPS, len(item_indices))
    )

    rows = []
    for k in K_VALUES:
        table = controls[controls.k == k]
        item_means = (
            table.groupby(["item_index", "random_index"])
            .delta_logit_contrast.mean()
            .groupby("item_index").mean()
            .reindex(item_indices).to_numpy()
        )
        bootstrap_means = item_means[draws].mean(axis=1)
        lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
        rows.append({
            "model_id": MODEL_ID,
            "layer": layer,
            "k": k,
            "n_examples": table[["item_index", "image_word", "condition"]]
                .drop_duplicates().shape[0],
            "mean_delta_logit_contrast": table.delta_logit_contrast.mean(),
            "mean_valid_digit_mass": table.valid_digit_mass.mean(),
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
        })
    summary = pd.DataFrame(rows)
    results_dir = ROOT_DIR / "steering/results" / MODEL_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "random_control_summary.csv"
    summary.to_csv(output_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved {output_path}", flush=True)
    return summary


def process_all():
    """Run random orthogonal controls for all folds at the peak RR layer only."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen bfloat16 control run")
    layer = get_peak_layer()
    print(f"Peak RR decoding layer: {layer}", flush=True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        local_files_only=True,
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_NAME, local_files_only=True)
    rating_token_ids = get_rating_token_ids(processor)
    with (ROOT_DIR / "stimuli/sentences.json").open(encoding="utf-8") as file:
        stimuli = json.load(file)

    controls, baselines = [], []
    for fold in FOLDS:
        control, baseline = process_fold(
            model, processor, stimuli, rating_token_ids, fold, layer
        )
        control["fold"] = fold
        baseline["fold"] = fold
        controls.append(control)
        baselines.append(baseline)
        print(f"Saved random controls for fold {fold}, layer {layer}", flush=True)
    return summarize_random_controls(controls, baselines, layer)


if __name__ == "__main__":
    process_all()

"""Save baseline and probe-steered rating logits for held-out stimuli.

The prompt is identical to the instruction-conditioned behavioral prompt. For
each fold and layer, steering changes only the final prompt token by
``k * s_train * scaler.scale_ * w_hat``. Layers before the final layer are
intervened on at the corresponding decoder-block output; the final layer is
intervened on after the model's final RMSNorm, matching the representation
saved during hidden-state extraction. Outputs are saved beside the probe under
``models/<model_id>/steering/layer<layer>/fold<fold>``.
"""

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_ID = "qwen3_vl_8b_instruct"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
LAYERS = None # list of layer indices to process; None means all decoder layers of this model
K_VALUES = [-2, -1, 0, 1, 2]
FOLDS = range(1, 11)
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
from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

from qwen_prompt import build_rating_inputs

if LAYERS is None:
    # get all decoder-layer indices from the locally cached model config
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    text_config = getattr(config, "text_config", config)
    LAYERS = list(range(text_config.num_hidden_layers))

def get_prompt(processor, item_index, image_word, condition):
    """Build the exact instruction-conditioned behavioral prompt for one item."""
    with (ROOT_DIR / "stimuli/sentences.json").open(encoding="utf-8") as file:
        stimulus = json.load(file)[int(item_index)]
    if condition not in {"congruent", "incongruent"}:
        raise ValueError(f"Unknown condition: {condition}")
    if image_word not in stimulus["word_options"]:
        raise ValueError(f"{image_word!r} is not an option for stimulus {item_index}")
    if condition == "congruent":
        final_word = image_word
    else:
        final_word = next(word for word in stimulus["word_options"] if word != image_word)
    sentence = f"{stimulus['sentence_frame']} {final_word}."
    image_path = ROOT_DIR / "stimuli/pictures" / f"{image_word}.png"
    with Image.open(image_path) as image:
        return build_rating_inputs(processor, image.convert("RGB"), sentence)


def _get_rating_token_ids(processor):
    """Return vocabulary IDs for the five single-token rating responses."""
    rating_ids = []
    for rating in range(1, 6):
        ids = processor.tokenizer.encode(str(rating), add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Rating {rating} is not a single token: {ids}")
        rating_ids.append(ids[0])
    return rating_ids


def _summarize_logits(logits, rating_token_ids):
    """Extract rating logits, their vocabulary mass, and high-versus-low contrast."""
    rating_logits = logits[rating_token_ids].float()
    valid_digit_mass = logits.float().softmax(dim=-1)[rating_token_ids].sum().item()
    values = rating_logits.cpu().tolist()
    row = {f"l{rating}": value for rating, value in enumerate(values, 1)}
    row["valid_digit_mass"] = valid_digit_mass
    row["logit_contrast"] = (row["l4"] + row["l5"] - row["l1"] - row["l2"]) / 2
    return row


def get_baseline_logits(model, inputs, rating_token_ids):
    """Run an unmodified forward pass and return rating-logit metrics."""
    inputs = inputs.to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs, return_dict=True, use_cache=False, logits_to_keep=1)
    logits = outputs.logits[0, -1, :]
    return _summarize_logits(logits, rating_token_ids)


def _repeat_inputs(inputs, count):
    """Repeat one unpadded multimodal prompt into a batch of identical prompts."""
    return {
        key: value.repeat((count,) + (1,) * (value.ndim - 1))
        for key, value in inputs.items()
    }


def _load_steering_step(fold, layer):
    """Return ``s_train * scaler.scale_ * w_hat`` for one fold and layer."""
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
        raise ValueError(f"Invalid training projection SD for fold {fold}, layer {layer}")
    return s_train * scaler.scale_ * w_hat


def get_steered_logits(model, inputs, fold, layer, k_list, rating_token_ids):
    """Return rating-logit metrics after each requested probe-direction intervention.

    All ``k`` conditions are evaluated in one batch. The intervention at the
    selected representation is ``k * s_train * scaler.scale_ * w_hat`` and is
    applied only to the final prompt-token hidden state. The final layer uses
    the post-RMSNorm output because that is the space used to train its probe.
    """
    k_values = np.asarray(k_list, dtype=np.float64)
    if k_values.ndim != 1 or k_values.size == 0 or not np.isfinite(k_values).all():
        raise ValueError("k_list must be a nonempty sequence of finite values")
    decoder = model.model.language_model
    if layer < 0 or layer >= len(decoder.layers):
        raise ValueError(f"Layer {layer} is outside 0..{len(decoder.layers) - 1}")
    intervention_module = (
        decoder.norm if layer == len(decoder.layers) - 1 else decoder.layers[layer]
    )

    unit_step = _load_steering_step(fold, layer)
    deltas = torch.as_tensor(
        k_values[:, None] * unit_step[None, :], device=model.device, dtype=torch.float32
    )
    batched_inputs = _repeat_inputs(inputs.to(model.device), len(k_values))
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

    handle = intervention_module.register_forward_hook(steer_final_token)
    try:
        with torch.inference_mode():
            outputs = model(
                **batched_inputs, return_dict=True, use_cache=False, logits_to_keep=1
            )
    finally:
        handle.remove()
    if len(hook_calls) != 1:
        raise AssertionError(f"Expected one hook call, observed {len(hook_calls)}")

    logits = outputs.logits[:, -1, :]
    return [
        {"k": float(k), **_summarize_logits(row_logits, rating_token_ids)}
        for k, row_logits in zip(k_values, logits, strict=True)
    ]


def _held_out_names(fold):
    """Return the 32 held-out filenames for one cross-validation fold."""
    with (ROOT_DIR / "decoding/cv/cv_lookup_pt.json").open(encoding="utf-8") as file:
        split = json.load(file)[str(fold)]
    if set(split["train"]) & set(split["test"]):
        raise ValueError(f"Training and test stimulus sets overlap in fold {fold}")
    return [name for names in split["test"].values() for name in names]


def _save_fold_layer(model, rating_token_ids, fold, layer, k_list, baselines):
    """Save baseline and steered CSVs for one fold/layer combination."""
    fold_dir = (
        ROOT_DIR / "models" / MODEL_ID / "steering" / f"layer{layer}" / f"fold{fold}"
    )
    steered_rows = []
    for metadata, inputs, _baseline_metrics in baselines.values():
        for metrics in get_steered_logits(
            model, inputs, fold, layer, k_list, rating_token_ids
        ):
            steered_rows.append({**metadata, **metrics})

    baseline_columns = [
        "item_index", "image_word", "condition", "valid_digit_mass",
        "l1", "l2", "l3", "l4", "l5", "logit_contrast",
    ]
    steered_columns = [
        "item_index", "image_word", "condition", "k", "valid_digit_mass",
        "l1", "l2", "l3", "l4", "l5", "logit_contrast",
    ]
    baseline_table = pd.DataFrame(
        [{**metadata, **metrics} for metadata, _inputs, metrics in baselines.values()],
        columns=baseline_columns,
    )
    steered_table = pd.DataFrame(steered_rows, columns=steered_columns)
    baseline_table.to_csv(fold_dir / "baseline_logits.csv", index=False)
    steered_table.to_csv(fold_dir / "steered_logits.csv", index=False)
    return baseline_table, steered_table


def _bootstrap_summary(combined, layers):
    """Summarize steering and bootstrap held-out stimulus sets within each layer."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for layer in layers:
        table = combined[combined.layer == layer]
        item_indices = np.sort(table.item_index.unique())
        if len(item_indices) != 80:
            raise ValueError(f"Expected 80 held-out stimulus sets at layer {layer}")
        draws = rng.integers(0, len(item_indices), (N_BOOTSTRAPS, len(item_indices)))
        for k in sorted(table.k.unique()):
            condition = table[table.k == k]
            set_means = (
                condition.groupby("item_index").delta_logit_contrast.mean()
                .reindex(item_indices).to_numpy()
            )
            bootstrap_means = set_means[draws].mean(axis=1)
            lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
            rows.append({
                "model_id": MODEL_ID,
                "layer": layer,
                "k": k,
                "n_examples": len(condition),
                "mean_delta_logit_contrast": condition.delta_logit_contrast.mean(),
                "mean_valid_digit_mass": condition.valid_digit_mass.mean(),
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
            })
    return pd.DataFrame(rows)


def process_all(layers=None, k_list=None):
    """Process every fold for selected layers and save the aggregate summary."""
    layers = LAYERS if layers is None else [int(layer) for layer in layers]
    k_list = K_VALUES if k_list is None else list(k_list)
    if not layers:
        raise ValueError("At least one layer is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Qwen bfloat16 steering run")

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
    rating_token_ids = _get_rating_token_ids(processor)

    combined_tables = []
    for fold in FOLDS:
        baselines = {}
        for name in _held_out_names(fold):
            item_index, image_word, condition = Path(name).stem.split("_")
            metadata = {
                "item_index": int(item_index),
                "image_word": image_word,
                "condition": condition,
            }
            inputs = get_prompt(processor, int(item_index), image_word, condition)
            metrics = get_baseline_logits(model, inputs, rating_token_ids)
            baselines[name] = (metadata, inputs, metrics)
        for layer in layers:
            baseline, steered = _save_fold_layer(
                model, rating_token_ids, fold, layer, k_list, baselines
            )
            merged = steered.merge(
                baseline[["item_index", "image_word", "condition", "logit_contrast"]],
                on=["item_index", "image_word", "condition"],
                how="left",
                validate="many_to_one",
                suffixes=("", "_baseline"),
            )
            merged["delta_logit_contrast"] = (
                merged.logit_contrast - merged.logit_contrast_baseline
            )
            merged["fold"] = fold
            merged["layer"] = layer
            combined_tables.append(merged)
            print(f"Saved fold {fold}, layer {layer}", flush=True)

    combined = pd.concat(combined_tables, ignore_index=True)
    keys = ["item_index", "image_word", "condition", "layer", "k"]
    if combined.duplicated(keys).any():
        raise AssertionError("Held-out fold outputs overlap")
    summary = _bootstrap_summary(combined, layers)
    results_dir = ROOT_DIR / "steering" / "results" / MODEL_ID
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = results_dir / "steering_summary.csv"
    if summary_path.is_file():
        previous = pd.read_csv(summary_path)
        previous = previous[~previous.layer.isin(layers)]
        summary = (
            pd.concat([previous, summary], ignore_index=True)
            .sort_values(["layer", "k"])
            .reset_index(drop=True)
        )
    summary.to_csv(summary_path, index=False)
    print(summary[summary.layer.isin(layers)].to_string(index=False), flush=True)
    print(f"Saved {summary_path}", flush=True)
    return summary


if __name__ == "__main__":
    process_all()

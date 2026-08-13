## What this is

A research pipeline studying picture-sentence congruency judgments: for 80 sentence-frame stimuli, each
paired with one of two possible final words and one of two possible pictures, the project compares human
survey ratings against Qwen3-VL vision-language models' behavioral ratings and internal hidden-state
representations.

There is no package manifest, test suite, linter, or build system — this is a flat collection of analysis
scripts run manually, stage by stage.

## Environment

- Python 3.12 venv at `.venv` (created with `--system-site-packages`, so most heavy deps like `torch` /
  `transformers` resolve from the system site-packages, not from `.venv` itself).
- Secrets/config live in `.env` (`PRIME_API_KEY`, `HF_TOKEN`, `QUALTRICS_API_TOKEN`, `QUALTRICS_DATACENTER`,
  plus CUDA path overrides) and are loaded via `python-dotenv`'s `load_dotenv()` at the top of scripts that
  need them.
- GPU-touching scripts (`hidden_states/hs_qwen.py`, `behavioral/behavioral_qwen.py`) must call
  `prepare_cuda()` (from the local `prepare_cuda.py`, duplicated in both `hidden_states/` and `behavioral/`)
  **before torch is imported anywhere** — it sets `CUDA_HOME`/`LD_LIBRARY_PATH`/etc. and restricts to a
  single physical GPU unless `allow_multi_gpu=True` is passed (needed for models sharded with
  `device_map="auto"`).
- Every script hardcodes absolute paths rooted at `/Intern/Erin/picture_sentence_congruency/...` rather than
  using paths relative to the file — the repo is assumed to live at exactly this location.
- Scripts use flat local imports (e.g. `from behavioral_qwen import extract_ratings`) with no package
  `__init__.py`, so each script must be run with its own subdirectory as the working directory.

## Running the pipeline

There's no CLI; each stage's `*_main.py` is a driver script meant to be hand-edited (which `model_id`s to
run, how many permutations, etc.) and then executed directly:

```bash
source .venv/bin/activate

cd stimuli && python sentences.py          # (re)generate stimuli/sentences.json from the `sentences` list
cd models && python models_lookup.py       # (re)generate models/models_lookup.json from the `models_lookup` dict

cd hidden_states && python hs_qwen.py      # extract per-layer hidden states (edit model_id at call site)
cd behavioral && python behavioral_main.py # get VLM ratings + accuracy/correlation/health analysis
cd decoding && python lr_cv.py             # (re)generate CV fold splits in decoding/cv/
cd decoding && python lr_main.py           # train/bootstrap/permute/significance-test/plot decoding
cd alignment && python alignment_main.py   # regress human ratings on decoder congruent-probability
cd confidence && python confidence_VLM.py  # confidence/probability-mass plots
cd confidence && python confidence_humans.py
```

Survey generation/ingestion (`survey/qualtrics_generate_survey.py`,
`survey/qualtrics_survey_assignment.py`, `survey/qualtrics_upload_images.py`,
`survey/qualtrics_survey_results.py`) is a separate, mostly one-shot flow for building the Qualtrics `.qsf`
survey files, assigning stimuli to survey versions A–D, and parsing raw Qualtrics response CSVs down into
`survey/results/ratings.csv` — the human ground-truth ratings every other stage joins against.

## Pipeline architecture

The stages run in this dependency order, each writing into `models/{model_id}/<stage>/`:

1. **`stimuli/`** — `sentences.py` is the source of truth for the 80 experimental items (sentence frame +
   two candidate final words); it exports `sentences.json`. `stimuli/pictures/*.png` are named after the
   words they depict (one picture per word).
2. **`survey/`** → **`survey/results/ratings.csv`** — human congruency ratings (1–5) per
   `(item_index, image_word, condition)`, the ground truth used by `behavioral/`, `alignment/`, and
   `confidence/confidence_humans.py`.
3. **`models/models_lookup.json`** — the central `model_id -> {title, color}` registry every downstream
   script reads to know which models exist and how to label/color their plots.
4. **`hidden_states/hs_qwen.py`** — loads a Qwen3-VL model, feeds it each (picture, sentence) pair, and
   saves the last-token-before-period hidden state at every layer as one `.pt` tensor per stimulus/condition
   into `models/{model_id}/hidden_states/`.
5. **`behavioral/behavioral_qwen.py`** — prompts the same models for a 1–5 compatibility rating, computed as
   the probability-weighted expectation over the digit tokens' next-token logits (not free-text generation);
   saves `models/{model_id}/behavior/ratings.csv`. `behavioral_accuracy.py` / `behavioral_correlation.py` /
   `behavioral_health.py` compare these against human ratings.
6. **`decoding/`** — `lr_cv.py` builds 10-fold, item-level (not row-level) cross-validation splits over the
   80 stimuli. `lr_train.py` trains one logistic-regression classifier per layer per fold to decode
   congruent-vs-incongruent from the layer's hidden state, producing per-item predicted probabilities in
   `models/{model_id}/decoding/results.csv` and `layerwise_accuracy.csv`. `lr_bootstrap.py` /
   `lr_permutation.py` / `lr_significance.py` establish statistical significance of layerwise accuracy;
   `lr_visualize.py` plots it.
7. **`alignment/alignment_rep_bev.py`** — per layer, regresses human ratings on the decoder's (z-scored)
   congruent-probability via OLS with item-clustered SEs and BH-FDR correction across layers, to test
   whether the linearly-decodable congruency signal in the model's representations tracks human graded
   judgments; `alignment_visualize.py` plots the resulting coefficients.
8. **`confidence/`** — separate look at rating *confidence* (distance from the scale midpoint / max softmax
   probability) rather than accuracy, for humans and VLMs respectively.

## Naming conventions

- **`model_id`** strings (e.g. `qwen3_vl_30b_a3b_instruct`) key `models_lookup.json` and the
  `models/{model_id}/` output tree; dense variants (2b/4b/8b/32b) use `Qwen3VLForConditionalGeneration`,
  MoE variants (30b-a3b) use `Qwen3VLMoeForConditionalGeneration`. All models use a single GPU.
- **Hidden-state / stimulus filenames**: `{item_index}_{image_word}_{condition}.pt`, where `item_index` is
  the 0-based index into `sentences.json`, `image_word` is whichever of the item's two `word_options` was
  paired with the picture, and `condition` is `congruent` (image word == sentence's final word) or
  `incongruent` (they differ). This same triple (`item_index`, `image_word`, `condition`) is the join key
  used everywhere results are merged with human ratings.

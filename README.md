# Picture–sentence congruency

A research pipeline comparing human and vision-language model (VLM) judgments of picture–sentence compatibility, decoding congruency from model representations, and relating that internal signal to the models' explicit judgments.

This is a collection of Python scripts run stage by stage. There is no package manifest, automated test suite, linter, build system, or unified command-line interface.

## Experimental design

`stimuli/sentences.py` defines 80 sentence frames, each with two possible final words, and exports `stimuli/sentences.json`. Pictures in `stimuli/pictures/` are named after the words they depict.

Each item produces four picture–sentence combinations: two matching (congruent) and two mismatching (incongruent), for 320 examples per model. For example, “She needed to put everything in the” has endings `library` and `refrigerator`, each paired with both pictures. Congruency is the experimental label defined by whether the picture word equals the final word; compatibility is the graded 1–5 judgment collected from humans or models.

Four counterbalanced Qualtrics surveys (A–D) each contain one example from every item: 40 congruent and 40 incongruent. Across surveys, each item's four combinations appear exactly once. The current local `survey/results/ratings.csv` contains 5,090 individual ratings from 67 participant IDs, covering all 320 combinations. These counts describe the current data rather than requirements for future collections.

## Models and outputs

`models/models_lookup.py` exports `models/models_lookup.json`, the shared registry of model IDs, titles, plotting colors, families, architecture types, and sizes. It currently lists 42 variants:

| Family | Variants | Coverage |
| --- | ---: | --- |
| Qwen3-VL | 6 | Instruction-tuned dense and MoE models |
| InternVL3 | 12 | Six instruction-tuned and six base models |
| Perception-LM (PLM) | 3 | Model size |
| LLaVA-OneVision | 3 | Model size |
| GLM-4.6V | 2 | Dense and MoE models |
| Gemma3 | 6 | Three instruction-tuned and three base models |
| Gemma4 | 10 | Five instruction-tuned and five base models |

The current local output tree has hidden-state directories and decoding results for all 42 variants, and behavioral ratings and alignment coefficients for 28. Base variants currently have representation analyses but no behavioral rating outputs. File presence does not establish completeness or reliability. Aggregate tables may cover fewer models if they have not been regenerated.

Stages share the key `(item_index, image_word, condition)`. Per-model outputs live under `models/{model_id}/`; aggregate tables and figures generally live in each analysis directory's `results/` and `plots/` subdirectories.

## Pipeline

### Human survey

`survey/qualtrics_survey_assignment.py` builds and verifies counterbalanced assignments. `qualtrics_upload_images.py` and `qualtrics_generate_survey.py` support image upload and survey `.qsf` generation. `qualtrics_survey_results.py` parses response CSVs into participant-level ratings.

The parser writes `survey/results/ratings.csv`, the shared human-rating input used by downstream analyses. Human comparisons average ratings by the shared stimulus key.

### Behavioral judgments

`behavioral/behavioral_{qwen,intern,plm,llava,glm,gemma3,gemma4}.py` implement model-specific extraction. Models receive a picture, sentence, and compatibility instructions using a 1–5 scale. Ratings are expectations over probabilities normalized across the five rating-digit tokens, rather than parsed free-text responses. Some families use additional generation or answer-position handling; `qwen_big.py` and `glm_big.py` provide separate large-model workflows.

`models/{model_id}/behavior/ratings.csv` includes the expected `rating`, most likely digit (`top_one_token`), probabilities `p1`–`p5`, `max_probability`, and `total_probability_mass` on rating digits in the full-vocabulary distribution. The last field checks whether valid answers receive probability even when the normalized five-digit distribution looks confident.

Analyses include:

- `behavioral_accuracy.py`: classify `rating > 3` as congruent and `rating <= 3` as incongruent; report overall and condition-specific accuracy and confusion matrices.
- `behavioral_correlation.py`: Spearman correlation with mean human ratings and 10,000 item-resampling bootstrap confidence intervals.
- `behavioral_health.py`: rating distributions and per-model summaries.
- `behavioral_concordance.py`: within-item comparisons of congruent against incongruent ratings.
- `behavioral_grounding.py`: correlation between ratings for the same final word under matching and mismatching pictures, used as a grounding diagnostic.
- `behavioral_instruction.py`: mean and minimum probability mass assigned to valid rating tokens.

### Hidden-state extraction

`hidden_states/hs_{qwen,intern,plm,llava,glm,gemma3,gemma4}.py` feed each picture–sentence pair to its model **without the behavioral rating instruction**. They locate the last period token and extract the preceding token's representation, intended to be the final token of the critical word. Tokenization must preserve that assumption.

Each example is saved as `models/{model_id}/hidden_states/{item_index}_{image_word}_{condition}.pt`: a CPU float32 tensor of shape `(number_of_layers, hidden_size)`, excluding the embedding state. Extraction and behavioral prompting are separate forward passes with different inputs.

Gemma3, Gemma4, LLaVA, InternVL3, and Qwen3-VL also support `extract_hidden_states(model_id, instructions=True)`. This mode shares the exact behavioral rating prompt and chat-template construction, including `add_generation_prompt=True`, and extracts each layer at the final input position that predicts the first response token. No rating digit is appended. Tensors use the same format and filenames but are saved under `models/{model_id}/hidden_states_instruction/`. The default `instructions=False` retains critical-word extraction. For Gemma and InternVL3, instruction mode requires an instruction-tuned checkpoint. Decoding training accepts the same `instructions=True` option to read these new tensors and write `decoding_instruction/`.

GLM instruction extraction uses the behavioral generation call with `enable_thinking=False` and two generated tokens. It requires the first token to be `<|begin_of_box|>` and extracts generation step 1 hidden states at that box token, matching the position producing `gen.scores[1]` for the rating digit. Unexpected prefixes or nonfinite states stop extraction. Original critical-word extraction remains available with `instructions=False`. For `glm_106b_a12b`, set `GLM_HS_MULTI_GPU=1` and `CUDA_VISIBLE_DEVICES` to two idle GPU indices; the model is sharded across those two GPUs only. This model matches `behavioral/glm_big.py` by forcing the box marker then a rating digit, and additionally checks that raw rating logits are finite.

PLM instruction extraction shares `behavioral/plm_prompt.py` with behavioral evaluation: the system content is a plain string, the assistant generation prompt is added, and separately tokenized `Rating: ` (including its trailing space) is appended to input IDs and attention mask. It extracts each layer at the final prefix token, matching behavioral `outputs.logits[0, -1]`, and saves to `hidden_states_instruction`. PLM models run sequentially on one GPU.

### Layerwise decoding

`decoding/lr_cv.py` creates reproducible 10-fold splits of the 80 items (seed 42) in `decoding/cv/`. All four examples of an item stay together, so its sentence frame cannot occur in both training and test sets. Each fold uses 72 training items (288 examples) and eight test items (32 examples).

`lr_train.py` fits a separate `StandardScaler` plus logistic regression at each layer and fold (`C=0.01`, `solver="liblinear"`, `max_iter=10000`). Scaling is fitted on training data. Congruent is label 1. Held-out predictions produce per-model `decoding/results.csv`, including `probability_congruent`, and `decoding/layerwise_accuracy.csv`.

`lr_bootstrap.py` resamples whole items 10,000 times to estimate accuracy uncertainty without retraining. `lr_permutation.py` independently flips each item's labels with probability 0.5 and retrains all fold/layer classifiers. It checkpoints each permutation and saves layerwise null accuracies plus the maximum accuracy across layers for each permutation. `lr_significance.py` compares observed accuracies with this maximum-statistic null distribution to account for testing multiple layers within a model.

Plotting and summary scripts compare layer trajectories, model sizes, and base versus instruction-tuned variants.

### Representation–behavior alignment

`alignment/alignment_rep_bev.py` joins held-out decoder probabilities with **the same model's behavioral ratings**, not human ratings. At each layer it standardizes the congruent probability and fits:

```text
rating ~ probability_z + C(item_index)
```

The OLS model includes item fixed effects and standard errors clustered by item. Benjamini–Hochberg FDR correction is applied across layers within each model. Results are saved to `models/{model_id}/alignment/coefficients.csv`. This measures whether the linearly decodable congruency signal tracks the model's own explicit graded judgments. Human-rating input is commented out in the current implementation.

`alignment_main.py` produces per-model plots and an aggregate summary of peak and final-layer coefficients. Pair-generation and visualization scripts inspect selected model comparisons.

### Confidence

`confidence/confidence_VLM.py` plots maximum probability within the normalized five-digit distribution, separated by congruency condition. `confidence_humans.py` plots signed `rating - 3`, reversing the sign for incongruent examples: positive values support the assigned condition, negative values oppose it.

## Environment and execution

- The local Python 3.12 virtual environment is `.venv`, created with `--system-site-packages`. Heavy dependencies such as PyTorch and Transformers may resolve from system packages. Other dependencies include pandas, NumPy, Pillow, scikit-learn, SciPy, statsmodels, matplotlib, tqdm, and python-dotenv. Model-specific dependencies and memory requirements vary.
- `.env` holds credentials and configuration for services, model access, and CUDA overrides. Scripts that need it use `load_dotenv()`; it is excluded from Git.
- Paths are generally hardcoded under `/Intern/Erin/picture_sentence_congruency`.
- Drivers use flat sibling imports. Run them from their own stage directory.
- The duplicated `prepare_cuda.py` helpers in `behavioral/` and `hidden_states/` configure CUDA. With `allow_multi_gpu=False`, the helper selects the physical GPU with the most free memory. Multi-GPU workflows preserve configured visibility and use sharding such as `device_map="auto"`. Call the helper before importing PyTorch when configuring new GPU workflows; some existing scripts currently import PyTorch earlier. Check script-specific GPU settings before launching.
- Several scripts configure CUDA and load metadata at import time. Extraction functions should be called explicitly; `hs_qwen.py` no longer launches the 235B model on import supports single-GPU models by default. For 235B extraction, explicitly set `QWEN_HS_MULTI_GPU=1` and select three idle devices with `CUDA_VISIBLE_DEVICES` before importing the module.

From the repository root:

```bash
mkdir -p /Intern/Erin/tmp
export TMPDIR=/Intern/Erin/tmp TMP=/Intern/Erin/tmp TEMP=/Intern/Erin/tmp
source .venv/bin/activate
```

Run stages individually from the root:

```bash
(cd stimuli && python sentences.py)        # regenerate stimuli JSON, if needed
(cd models && python models_lookup.py)     # regenerate registry, if needed
(cd hidden_states && python hs_main.py)    # after selecting family/models
(cd behavioral && python behavioral_main.py)  # after restoring extractor import
(cd decoding && python lr_cv.py)           # regenerate item splits, if needed
(cd decoding && python lr_main.py)         # after enabling required stages/models
(cd alignment && python alignment_main.py) # requires decoding and behavioral outputs
(cd confidence && python confidence_VLM.py)
(cd confidence && python confidence_humans.py)
```

These commands are a workflow guide, not an unattended end-to-end runner. The current drivers need manual configuration:

- `hs_main.py` defaults to `plm_1b`, `plm_3b`, and `plm_8b` (or accepts PLM model IDs as command-line arguments) with `instructions=True`, completing extraction, `lr_main.py`, and `alignment_main.py` for that model before starting the next. Each stage runs in a separate process.
- `behavioral_main.py` selects Qwen models, but all `extract_ratings` imports are commented out. Restore the matching import before extracting ratings. Otherwise, its caught error skips per-model work and proceeds to aggregate analyses of existing files.
- `lr_main.py` trains and bootstraps instruction-conditioned decoding, then saves blue/pink comparison PNGs under `decoding/plots/{family}_instruction/`. Pass model IDs as arguments to process specific models; the default is all five instruction-tuned Gemma4 variants. It does not run permutations or significance testing.
- `alignment_main.py` computes instruction-conditioned coefficients and final-layer pairs, then saves comparison PNGs under `alignment/plots/{family}_instruction/`. Pass model IDs as arguments; the default is all five instruction-tuned Gemma4 variants. It requires behavioral ratings, instruction decoding results, and original alignment coefficients for comparison.

Some decoding stages skip work when output files exist, and permutation runs reuse seed checkpoints. When inputs or settings change, deliberately invalidate the relevant outputs before recomputation. The permutation runner currently requests 70 worker processes; adjust this to available CPU and memory resources.

## Reliability notes

`notes/` contains diagnostics and appendices for large GLM behavioral extraction. In particular, `appendix_glm46v_4gpu_reliability.md` documents NaN logits and unreliable answers in tested sharded GLM-4.6V configurations. Those notes do not establish a confirmed root cause or validate all saved GLM outputs. Consult the diagnostics before interpreting or reporting behavioral results from affected configurations. Large Qwen workflows also have script-specific multi-GPU handling and reliability comments.

## Naming conventions

- `model_id` keys the registry and the `models/{model_id}/` directory.
- `item_index` is the zero-based position in `stimuli/sentences.json`.
- `image_word` identifies the picture; `condition` is `congruent` or `incongruent`.
- Given an item's two endings, `image_word` and `condition` uniquely identify the final word, so it need not appear separately in the shared join key or hidden-state filename.
- Layer indices in decoding outputs are zero-based indices into the saved layer tensor; embedding states are excluded.

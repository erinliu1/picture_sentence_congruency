"""Summarize CLIP similarity using stimulus-set bootstrap and saved decoding folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'CLIP/results'
N_BOOTSTRAP = 10_000
SEED = 42


def summarize_similarity():
    df = pd.read_csv(RESULTS / 'similarity.csv')
    keys = ['item_index', 'image_word', 'condition']
    if len(df) != 320 or df.duplicated(keys).any():
        raise ValueError('Expected 320 unique stimulus pairs')
    if not np.isfinite(df[['clip_similarity', 'human_mean_rating']]).all().all():
        raise ValueError('Missing or nonfinite similarities or human ratings')
    counts = pd.crosstab(df.item_index, df.condition)
    if (counts.shape != (80, 2) or set(counts.columns) != {'congruent', 'incongruent'}
            or not counts.eq(2).all().all()):
        raise ValueError('Expected 80 stimulus sets with two pairs per condition')
    df = df.sort_values(keys).reset_index(drop=True)
    similarity = df.clip_similarity.to_numpy().reshape(80, 4)
    human = df.human_mean_rating.to_numpy().reshape(80, 4)
    congruent = df.condition.eq('congruent').to_numpy().reshape(80, 4)
    congruent_scores = similarity[congruent].reshape(80, 2)
    incongruent_scores = similarity[~congruent].reshape(80, 2)
    mean_congruent = congruent_scores.mean()
    mean_incongruent = incongruent_scores.mean()
    set_differences = congruent_scores.mean(axis=1) - incongruent_scores.mean(axis=1)
    concordant = congruent_scores.min(axis=1) > incongruent_scores.max(axis=1)

    # Sampling sets preserves all four pairs and repeats entire sets when selected twice.
    draws = np.random.default_rng(SEED).integers(0, 80, size=(N_BOOTSTRAP, 80))
    bootstrap_differences = set_differences[draws].mean(axis=1)
    difference_ci = np.percentile(bootstrap_differences, [2.5, 97.5])
    rho = spearmanr(similarity.ravel(), human.ravel()).statistic
    # Recompute ranks within each bootstrap sample, including ties from repeated sets.
    similarity_ranks = rankdata(similarity[draws].reshape(N_BOOTSTRAP, 320), axis=1)
    human_ranks = rankdata(human[draws].reshape(N_BOOTSTRAP, 320), axis=1)
    similarity_ranks -= similarity_ranks.mean(axis=1, keepdims=True)
    human_ranks -= human_ranks.mean(axis=1, keepdims=True)
    bootstrap_rhos = (similarity_ranks * human_ranks).sum(axis=1) / np.sqrt(
        (similarity_ranks ** 2).sum(axis=1) * (human_ranks ** 2).sum(axis=1))
    if not np.isfinite(bootstrap_rhos).all():
        raise ValueError('Undefined correlation in a bootstrap sample')
    rho_ci = np.percentile(bootstrap_rhos, [2.5, 97.5])

    with (ROOT / 'decoding/cv/cv_indices.json').open() as handle:
        folds = json.load(handle)
    if len(folds) != 10:
        raise ValueError('Expected ten saved decoding folds')
    all_sets = set(df.item_index)
    X = df[['clip_similarity']].to_numpy()
    y = df.condition.eq('congruent').astype(int).to_numpy()
    predictions = np.full(len(df), -1, dtype=int)
    probabilities = np.full(len(df), np.nan)
    fold_ids = np.zeros(len(df), dtype=int)
    for fold, split in folds.items():
        train_sets, test_sets = set(split['train']), set(split['test'])
        if (train_sets & test_sets or train_sets | test_sets != all_sets
                or len(train_sets) != 72 or len(test_sets) != 8):
            raise ValueError(f'Invalid stimulus-set split in fold {fold}')
        train = df.item_index.isin(train_sets).to_numpy()
        test = df.item_index.isin(test_sets).to_numpy()
        if (fold_ids[test] != 0).any():
            raise ValueError('Repeated held-out stimulus sets')
        classifier = make_pipeline(StandardScaler(), LogisticRegression(
            C=0.01, solver='liblinear', max_iter=10_000, random_state=SEED))
        classifier.fit(X[train], y[train])
        predictions[test] = classifier.predict(X[test])
        probabilities[test] = classifier.predict_proba(X[test])[:, 1]
        fold_ids[test] = int(fold)
    if (fold_ids == 0).any():
        raise ValueError('Some stimulus pairs were never held out')
    accuracy = np.mean(predictions == y)
    cv_results = df[keys].copy()
    cv_results['fold'] = fold_ids
    cv_results['label'] = y
    cv_results['predicted_label'] = predictions
    cv_results['probability_congruent'] = probabilities
    cv_results['correct'] = predictions == y
    cv_results.to_csv(RESULTS / 'similarity_cv_predictions.csv', index=False)

    summary = pd.DataFrame([{
        'mean_similarity_congruent': mean_congruent,
        'mean_similarity_incongruent': mean_incongruent,
        'mean_congruent_incongruent_similarity_difference': mean_congruent - mean_incongruent,
        'bootstrap_mean_congruent_incongruent_similarity_difference': bootstrap_differences.mean(),
        'difference_ci_lower': difference_ci[0],
        'difference_ci_upper': difference_ci[1],
        'concordant_sets': int(concordant.sum()),
        'within_set_concordance': concordant.mean(),
        'spearman_rho_human': rho,
        'spearman_rho_ci_lower': rho_ci[0],
        'spearman_rho_ci_upper': rho_ci[1],
        'cv_binary_congruency_accuracy': accuracy,
        'cv_correct_pairs': int((predictions == y).sum()),
        'n_stimulus_sets': 80,
        'n_pairs': len(df),
        'n_bootstrap': N_BOOTSTRAP,
        'bootstrap_seed': SEED,
        'ci_method': 'percentile',
        'n_cv_folds': len(folds),
        'logistic_regression_C': 0.01,
    }])
    summary.to_csv(RESULTS / 'similarity_summary.csv', index=False)
    print(summary.to_string(index=False))
    return summary


if __name__ == '__main__':
    summarize_similarity()

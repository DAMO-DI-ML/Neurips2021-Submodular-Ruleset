#!/usr/bin/env python3

from typing import Dict
from datautil import DatasetLoader, DATASETS
from features import FeatureBinarizer
import argparse
from functools import partial

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

from model import SubmodularRuleSet


pd.set_option("display.max_rows", None, "display.max_columns", None)


parser = argparse.ArgumentParser()
parser.add_argument('-c', dest='njobs', type=int, default=8, help='CV N jobs.')
parser.add_argument('-p', dest='parallelism', type=int, default=8, help='Parallelism.')
parser.add_argument('-o', dest='outer_nfold', type=int, default=5, help='Number of outer folds.')
parser.add_argument('-i', dest='inner_nfold', type=int, default=5, help='Number of inner folds.')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose.')
args = parser.parse_args()


SMALL_DATASETS = [
    'heart', 'pima', 'transfusion', 'ionosphere', 'ILPD'
]
LARGE_DATASETS = [
    'musk', 'gas'
]


def OneFold(X, *args, **kwargs):
    indices = np.arange(X.shape[0], dtype=int)
    yield indices, indices


def CustomKFold(X, k=3, frac=10, *args, **kwargs):
    n = X.shape[0]
    test_idx = np.arange(n, dtype=int)
    for i in range(k):
        perm = np.random.permutation(n)
        drop = n // frac
        train_idx = perm[:-drop]
        yield train_idx, test_idx


def best_model_idx(cv_results, main_metric='min_test_score'):
    df = pd.DataFrame(cv_results)
    scores = df.filter(regex=r'split\d+_test_score').to_numpy()
    median = np.median(scores, axis=-1)
    minimum = np.min(scores, axis=-1)
    df['median_test_score'] = median
    df['min_test_score'] = minimum
    df.sort_values(
        by=[
            main_metric,
            'std_test_score',
            'param_beta_complex',
            'param_beta_diverse',
            'param_max_num_rules',
        ],
        ascending=[False, True, False, False, True],
        inplace=True
    )
    idx = df.index[0]
    cv_results['params'][idx]['parallelism'] = 0
    return idx


def train(name, X, y):
    num_samples = X.shape[0]
    num_pos = np.sum(y).item()
    num_neg = num_samples - num_pos
    beta_pos = num_samples / num_pos
    beta_neg = num_samples / num_neg
    beta_neg = round(beta_neg / beta_pos, 3)
    learner = SubmodularRuleSet(
        beta_pos=1, beta_neg=1,
        parallelism=args.parallelism,
        warmcache=0, bestsubset=4,
        verbose=args.verbose
    )

    small_grid = {
        'parallelism': [args.parallelism],
        'max_num_rules': [16, 8],
        'beta_diverse': [0.01, 0.1, 0.5],
        'beta_complex': [0.1, 1, 2, 4, 8, 16]
    }
    large_grid = {
        'parallelism': [args.parallelism],
        'max_num_rules': [32, 16],
        'beta_diverse': [0.01, 0.1, 0.5],
        'beta_complex': [1, 4, 8, 16, 32]
    }
    large_x_grid = {
        'parallelism': [args.parallelism],
        'max_num_rules': [16],
        'beta_diverse': [0.01, 0.5],
        'beta_complex': [1, 8, 32]
    }

    if name in SMALL_DATASETS:
        grid = small_grid
        cv = StratifiedKFold(10, shuffle=True, random_state=42)
        refit = partial(best_model_idx, main_metric='mean_test_score')
    elif name in LARGE_DATASETS:
        grid = large_x_grid
        cv = StratifiedKFold(3, shuffle=True, random_state=42)
        refit = partial(best_model_idx, main_metric='min_test_score')
        learner.warmcache = 0
    elif X.shape[0] < 5000:
        grid = small_grid
        cv = CustomKFold(X, k=args.inner_nfold, frac=10)
        refit = partial(best_model_idx, main_metric='min_test_score')
    else:
        grid = large_grid
        cv = CustomKFold(X, k=args.inner_nfold, frac=10)
        refit = partial(best_model_idx, main_metric='min_test_score')

    clf = GridSearchCV(
        estimator=learner, param_grid=grid,
        cv=cv, scoring='accuracy', n_jobs=args.njobs,
        verbose=3, refit=refit
    )
    clf.fit(X, y)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf


def test(model, X, y):
    y_hat = model.predict(X)
    acc = accuracy_score(y, y_hat)
    itemsets = model.best_estimator_.itemsets
    overlap = model.best_estimator_.overlap_on(X)
    return acc, itemsets, overlap



def process(name: str):
    summary = pd.DataFrame()
    summary['dataset'] = (name,)
    best_params, best_models = [], []

    df = DatasetLoader(name).dataframe
    summary['samples'] = df.shape[0]
    summary['features'] = df.shape[1] - 1

    # Separate target variable
    y = df.pop('label')

    # Binarize the features
    binarizer = FeatureBinarizer(numThresh=9, negations=True, threshStr=True)
    df = binarizer.fit_transform(df)
    df.columns = [' '.join(col).strip() for col in df.columns.values]

    X, y = df.to_numpy(), y.to_numpy()
    summary['binarized'] = X.shape[1]
    summary['positives'] = np.sum(y)
    summary['negatives'] = np.sum(1 - y)

    # K-fold CV using fixed random seed
    skf = StratifiedKFold(args.outer_nfold, shuffle=True, random_state=42)
    accuracies, nrules, nliterals, overlaps = [], [], [], []
    fold = 0
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print('---> %s: CV fold %d ...' % (name, fold))
        model = train(name, X_train, y_train)

        if hasattr(model, 'best_params_'):
            best_params.append(model.best_params_)
        if hasattr(model, 'best_estimator_'):
            best_models.append(model.best_estimator_.itemsets)

        acc, itemsets, overlap = test(model, X_test, y_test)
        accuracies.append(acc)
        nrules.append(len(itemsets))
        nliterals.append(np.sum([len(itemset) for itemset in itemsets]))
        overlaps.append(overlap)

        print()
        print('---> %s: CV fold %d Done.' % (name, fold))
        print()
        print('Itemsets:', itemsets)
        print('Accuracy:', accuracies)
        print('NumRules:', nrules)
        print('NumLiterals:', nliterals)
        print('Overlaps:', overlaps)
        fold += 1

    summary['acc'] = np.mean(accuracies)
    summary['acc_std'] = np.std(accuracies, ddof=1)
    summary['rule'] = np.mean(nrules)
    summary['rule_std'] = np.std(nrules, ddof=1)
    summary['literal'] = np.mean(nliterals)
    summary['literal_std'] = np.std(nliterals, ddof=1)
    complexities = np.add(nrules, nliterals)
    summary['complexity'] = np.mean(complexities)
    summary['complexity_std'] = np.std(complexities, ddof=1)
    summary['overlap'] = np.mean(overlaps)
    summary['overlap_std'] = np.std(overlaps, ddof=1)

    print(summary.head(1))
    return summary, best_params, best_models

def main():
    summaries = []
    for name in DATASETS:
        summary, best_params, best_models = process(name)
        summaries.append(summary)

        print()
        print('Best params for', name)
        print()
        print(best_params)

        print()
        print('Best itemsets for', name)
        print()
        print(best_models)

        tmp = pd.concat(summaries)
        print(tmp)

    summaries = pd.concat(summaries)
    print(summaries)


if __name__ == '__main__':
    main()

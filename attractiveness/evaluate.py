"""Evaluates ridge regression on top of frozen features: either on a fixed train/test split or with k-fold cross-validation.

SCUT-FBP5500, the official 60/40 split:
  python -m attractiveness.evaluate --features features\\scut_clip_vit_l14.npz --ratings <SCUT>\\train_test_files\\All_labels.txt
      --train-list "<SCUT>\\train_test_files\\split_of_60%training and 40%testing\\train.txt"
      --test-list "<SCUT>\\train_test_files\\split_of_60%training and 40%testing\\test.txt"
Personal ratings, cross-validation, plus how well an already trained (e.g. SCUT) model agrees with them:
  python -m attractiveness.evaluate --features features\\personal_clip_vit_l14.npz --ratings personal_ratings.csv --reference-model models\\scut_clip_vit_l14.joblib
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold

from attractiveness.regression import load_features, load_ratings, make_model, metrics, select


def _format(name, result):
    return f"{name:<45} Pearson {result['pearson']:.4f}   MAE {result['mae']:.4f}   RMSE {result['rmse']:.4f}"


def _evaluate_split(names, features, train_list, test_list):
    _, train_x, train_y = select(names, features, load_ratings(train_list))
    _, test_x, test_y = select(names, features, load_ratings(test_list))
    model = make_model().fit(train_x, train_y)
    return metrics(model.predict(test_x), test_y), len(train_y), len(test_y)


def _evaluate_cross_validation(names, features, ratings, folds, repeats):
    _, x, y = select(names, features, ratings)
    results = []
    for repeat in range(repeats):
        predicted = np.empty_like(y)
        for train, test in KFold(folds, shuffle=True, random_state=repeat).split(x):
            predicted[test] = make_model().fit(x[train], y[train]).predict(x[test])
        results.append(metrics(predicted, y))
    return {key: float(np.mean([result[key] for result in results])) for key in results[0]}, len(y)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True, nargs="+", type=Path)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--test-list", type=Path)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-model", type=Path, help="An already trained model to compare against")
    args = parser.parse_args()

    ratings = load_ratings(args.ratings)
    candidates = [[path] for path in args.features] + ([args.features] if len(args.features) > 1 else [])
    for paths in candidates:
        names, features, backbones = load_features(paths)
        label = " + ".join(backbones)
        if args.train_list and args.test_list:
            result, train_size, test_size = _evaluate_split(names, features, args.train_list, args.test_list)
            print(_format(f"{label} (train {train_size}, test {test_size})", result))
        else:
            result, size = _evaluate_cross_validation(names, features, ratings, args.folds, args.repeats)
            print(_format(f"{label} ({args.folds}-fold CV, {size} faces)", result))

    if args.reference_model:
        reference = joblib.load(args.reference_model)
        by_backbone = {str(np.load(path)["backbone"]): path for path in args.features}
        names, features, _ = load_features([by_backbone[backbone] for backbone in reference["backbones"]])
        _, x, y = select(names, features, ratings)
        if args.test_list:
            _, x, y = select(names, features, load_ratings(args.test_list))
        print(_format(f"{args.reference_model.name} as is ({len(y)} faces)", metrics(reference["model"].predict(x), y)))


if __name__ == "__main__":
    main()

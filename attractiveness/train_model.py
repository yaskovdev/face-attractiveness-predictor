"""Trains the final ridge regression on all rated faces and saves it.

Example: python -m attractiveness.train_model --features features\\scut_clip_vit_l14.npz --ratings <SCUT>\\train_test_files\\All_labels.txt --output models\\scut_clip_vit_l14.joblib
"""
import argparse
from pathlib import Path

import joblib

from attractiveness.regression import load_features, load_ratings, make_model, select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True, nargs="+", type=Path)
    parser.add_argument("--ratings", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    names, features, backbones = load_features(args.features)
    _, x, y = select(names, features, load_ratings(args.ratings))
    model = make_model().fit(x, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"backbones": backbones, "model": model}, args.output)
    print(f"Trained on {len(y)} faces with {' + '.join(backbones)}, alpha {model[-1].alpha_:g}; saved to {args.output}")


if __name__ == "__main__":
    main()

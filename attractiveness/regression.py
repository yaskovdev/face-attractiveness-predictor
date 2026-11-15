"""Loading of features and ratings, the regression model and evaluation metrics."""
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_features(paths):
    """Loads one or more feature files and concatenates the features of the faces present in all of them."""
    loaded = [np.load(path) for path in paths]
    common = set(loaded[0]["names"])
    for data in loaded[1:]:
        common &= set(data["names"])
    names = sorted(common)
    blocks = []
    for data in loaded:
        index = {name: i for i, name in enumerate(data["names"])}
        blocks.append(data["features"][[index[name] for name in names]])
    return names, np.concatenate(blocks, axis=1), [str(data["backbone"]) for data in loaded]


def load_ratings(path):
    """Reads `image score` lines (SCUT-FBP5500 .txt files) or an `image,score` CSV; keys are file names without extension."""
    path = Path(path)
    ratings = {}
    with path.open(encoding="utf-8") as file:
        if path.suffix.lower() == ".csv":
            for row in csv.DictReader(file):
                if row["score"] != "":
                    ratings[Path(row["image"]).stem] = float(row["score"])
        else:
            for line in file:
                if line.strip():
                    name, score = line.split()
                    ratings[Path(name).stem] = float(score)
    return ratings


def select(names, features, ratings):
    """Keeps only the faces that have both features and a rating."""
    rows = [i for i, name in enumerate(names) if name in ratings]
    return [names[i] for i in rows], features[rows], np.array([ratings[names[i]] for i in rows])


def make_model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 5, 26)))


def metrics(predicted, expected):
    return {
        "pearson": float(np.corrcoef(predicted, expected)[0, 1]),
        "mae": float(np.mean(np.abs(predicted - expected))),
        "rmse": float(np.sqrt(np.mean((predicted - expected) ** 2))),
    }

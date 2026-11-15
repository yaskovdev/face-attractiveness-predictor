"""Predicts the attractiveness score of the largest face on each given photo.

Example: python -m attractiveness.predict --model models\\scut_clip_vit_l14.joblib photo1.jpg photo2.jpg --save-aligned aligned_output
The aligned crops show exactly what the model sees; check them if a score looks suspicious.
"""
import argparse
from pathlib import Path

import cv2
import joblib
import numpy as np

from attractiveness.face_alignment import FaceAligner, read_image
from attractiveness.feature_extractors import FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--save-aligned", type=Path)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    saved = joblib.load(args.model)
    aligner = FaceAligner()
    faces, paths = [], []
    for path in args.images:
        image = read_image(path)
        face = aligner.align(image) if image is not None else None
        if face is None:
            print(f"{path}: no face found")
            continue
        faces.append(face)
        paths.append(path)
        if args.save_aligned:
            args.save_aligned.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", face)[1].tofile(str(args.save_aligned / (path.stem + ".png")))
    if not faces:
        return

    features = np.concatenate([FeatureExtractor(backbone).extract(faces) for backbone in saved["backbones"]], axis=1)
    for path, score in zip(paths, saved["model"].predict(features)):
        print(f"{path}: {score:.2f}")


if __name__ == "__main__":
    main()

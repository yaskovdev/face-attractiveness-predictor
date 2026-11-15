"""Extracts features of all aligned faces in a directory with a frozen backbone and saves them to an .npz file.

Example: python -m attractiveness.extract_features --images C:\\dev\\fbp_data\\aligned\\scut --backbone clip_vit_l14 --output features\\scut_clip_vit_l14.npz
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from attractiveness.face_alignment import read_image
from attractiveness.feature_extractors import BACKBONES, FeatureExtractor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path, help="Directory with aligned faces (see align_images)")
    parser.add_argument("--backbone", required=True, choices=BACKBONES)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chunk", type=int, default=256)
    args = parser.parse_args()

    extractor = FeatureExtractor(args.backbone)
    paths = sorted(args.images.glob("*.png"))
    features = []
    for start in tqdm(range(0, len(paths), args.chunk)):
        faces = [read_image(path) for path in paths[start:start + args.chunk]]
        features.append(extractor.extract(faces))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, names=np.array([path.stem for path in paths]), features=np.concatenate(features), backbone=args.backbone)
    print(f"Saved features of {len(paths)} faces to {args.output}")


if __name__ == "__main__":
    main()

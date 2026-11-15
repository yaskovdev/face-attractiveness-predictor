"""Aligns every face photo in a directory and writes the aligned crops as PNG files with the same base names.

Example: python -m attractiveness.align_images --input C:\\dev\\fbp_data\\SCUT-FBP5500_v2.1\\SCUT-FBP5500_v2\\Images --output C:\\dev\\fbp_data\\aligned\\scut
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from attractiveness.face_alignment import FaceAligner, read_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    aligner = FaceAligner()
    paths = sorted(path for path in args.input.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    failed = []
    for path in tqdm(paths):
        image = read_image(path)
        aligned = aligner.align(image) if image is not None else None
        if aligned is None:
            failed.append(path.name)
            continue
        cv2.imencode(".png", aligned)[1].tofile(str(args.output / (path.stem + ".png")))
    print(f"Aligned {len(paths) - len(failed)} of {len(paths)} images.")
    if failed:
        print("No face found in:", ", ".join(failed))


if __name__ == "__main__":
    main()

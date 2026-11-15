"""Shows photos one by one and records your attractiveness rating of the face on each of them.

Example: python -m personal_ratings.rate_photos --images C:\\dev\\fbp_data\\photofeeler_images --output C:\\dev\\fbp_data\\personal_ratings.csv

Keys: 1-9 and 0 (= 10) rate the face on a 1-10 scale, S skips a photo that cannot be rated (no face, several people,
too blurry), B goes back to the previous photo, Q or Esc quits. Progress is saved after every key press, so rating can
be interrupted and resumed at any time. Photos are shown in a fixed random order to avoid order effects.
"""
import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np

from attractiveness.face_alignment import read_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
WINDOW = "Rate the face: 1-9, 0 = 10, S = skip, B = back, Q = quit"
DISPLAY_HEIGHT = 800
DISPLAY_WIDTH = 1400


def _load(path):
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as file:
        return {row["image"]: row for row in csv.DictReader(file)}


def _save(path, ratings):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image", "score", "rated_at"])
        writer.writeheader()
        writer.writerows(ratings.values())
    temporary.replace(path)


def _render(image, text):
    scale = min(DISPLAY_HEIGHT / image.shape[0], DISPLAY_WIDTH / image.shape[1])
    shown = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    bar = np.full((40, shown.shape[1], 3), 32, dtype=np.uint8)
    cv2.putText(bar, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([shown, bar])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    random.Random(42).shuffle(paths)
    ratings = _load(args.output)
    history = []
    position = next((i for i, path in enumerate(paths) if path.name not in ratings), len(paths))

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    while position < len(paths):
        path = paths[position]
        image = read_image(path)
        if image is None:
            position += 1
            continue
        previous = ratings.get(path.name, {}).get("score")
        status = f"{len(ratings)}/{len(paths)} rated" + (f", current: {previous or 'skipped'}" if path.name in ratings else "")
        cv2.imshow(WINDOW, _render(image, status))
        key = cv2.waitKey(0) & 0xFF
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("b"), ord("B"), 8):
            if history:
                position = history.pop()
            continue
        if ord("0") <= key <= ord("9"):
            score = str(10 if key == ord("0") else key - ord("0"))
        elif key in (ord("s"), ord("S")):
            score = ""
        else:
            continue
        ratings[path.name] = {"image": path.name, "score": score, "rated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save(args.output, ratings)
        history.append(position)
        position = next((i for i in range(position + 1, len(paths)) if paths[i].name not in ratings), len(paths))
    cv2.destroyAllWindows()
    print(f"{sum(1 for row in ratings.values() if row['score'])} photos rated, saved to {args.output}")


if __name__ == "__main__":
    main()

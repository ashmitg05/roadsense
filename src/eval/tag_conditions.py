"""Hand-tag a random sample of test images by condition (keyboard, OpenCV window).

Keys: 1 day/dry  2 day/wet  3 night/dry  4 night/wet  s skip  q quit.
Appends to data/conditions.csv (image,lighting,surface); already-tagged images are skipped.

    python -m src.eval.tag_conditions --n 150
"""

import argparse
import csv
import random
from pathlib import Path

import cv2

KEYS = {
    ord("1"): ("day", "dry"),
    ord("2"): ("day", "wet"),
    ord("3"): ("night", "dry"),
    ord("4"): ("night", "wet"),
}


def main(img_list: Path, out: Path, n: int, seed: int) -> None:
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["image"] for r in csv.DictReader(f)}
    imgs = [Path(p) for p in img_list.read_text().splitlines() if p.strip()]
    todo = [p for p in random.Random(seed).sample(imgs, min(n, len(imgs))) if p.name not in done]
    new = not out.exists()
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["image", "lighting", "surface"])
        for i, p in enumerate(todo):
            cv2.imshow("1 day/dry  2 day/wet  3 night/dry  4 night/wet  s skip  q quit", cv2.imread(str(p)))
            cv2.setWindowTitle(
                "1 day/dry  2 day/wet  3 night/dry  4 night/wet  s skip  q quit", f"{i + 1}/{len(todo)}"
            )
            while (k := cv2.waitKey(0) & 0xFF) not in (*KEYS, ord("s"), ord("q")):
                pass
            if k == ord("q"):
                break
            if k in KEYS:
                w.writerow([p.name, *KEYS[k]])
                f.flush()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", type=Path, default=Path("data/splits/test.txt"))
    ap.add_argument("--out", type=Path, default=Path("data/conditions.csv"))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    main(a.list, a.out, a.n, a.seed)

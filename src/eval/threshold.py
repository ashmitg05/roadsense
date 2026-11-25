"""Targeted fix for the top failure mode (missed potholes at the default conf 0.25):
pick a per-class confidence threshold that maximises F1 on VAL, then report test
precision/recall/F1 at the global 0.25 vs the tuned thresholds. Test never chooses.

    python -m src.eval.threshold models/exp_v8s/model_int8_head.onnx --name exp_v8s_int8head
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.data.convert import NAMES
from src.eval.analyze import load_gt, predictor
from src.eval.metrics import match

GRID = np.round(np.arange(0.05, 0.71, 0.05), 2)


def collect(predict, img_list: Path) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per image: preds (N,6), tp@IoU0.5 (N,), gt (M,5)."""
    out = []
    for p in [Path(x) for x in img_list.read_text().splitlines() if x.strip()]:
        img = cv2.imread(str(p))
        gt = load_gt(p, img.shape[1], img.shape[0])
        pred = predict(img)
        tp = match(pred, gt)[:, 0] if len(pred) else np.zeros(0, bool)
        out.append((pred, tp, gt))
    return out


def prf(data, thr: dict[int, float]) -> dict:
    res = {}
    for c, name in enumerate(NAMES):
        tp = fp = n_gt = 0
        for pred, t, gt in data:
            keep = (pred[:, 5] == c) & (pred[:, 4] >= thr[c]) if len(pred) else np.zeros(0, bool)
            tp += int(t[keep].sum())
            fp += int((~t[keep]).sum())
            n_gt += int((gt[:, 0] == c).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / n_gt if n_gt else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        res[name] = {
            "thr": thr[c],
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "n_gt": n_gt,
        }
    return res


def main(weights: str, name: str, imgsz: int) -> dict:
    predict = predictor(weights, imgsz)
    val = collect(predict, Path("data/splits/val.txt"))
    test = collect(predict, Path("data/splits/test.txt"))
    tuned = {}
    for c, n in enumerate(NAMES):
        f1s = [prf(val, dict.fromkeys(range(len(NAMES)), t))[n]["f1"] for t in GRID]
        tuned[c] = float(GRID[int(np.argmax(f1s))])
    default = dict.fromkeys(range(len(NAMES)), 0.25)
    report = {
        "weights": weights,
        "chosen_on": "val",
        "thresholds": {NAMES[c]: t for c, t in tuned.items()},
        "test_default_0.25": prf(test, default),
        "test_tuned": prf(test, tuned),
    }
    Path(f"results/threshold_{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    main(a.weights, a.name, a.imgsz)

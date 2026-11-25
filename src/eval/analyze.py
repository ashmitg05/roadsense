"""Error analysis: metrics by condition, box size and class; FP/FN breakdown; failure gallery.

Conditions come from `data/conditions.csv` (image,lighting,surface) when present;
otherwise lighting is auto-tagged night/dim/bright from mean brightness and surface is
"unknown". Tag a sample by hand with `src.eval.tag_conditions` for wet/dry.

    python -m src.eval.analyze runs/baseline/weights/best.pt --list data/splits/test.txt --name baseline
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.data.convert import NAMES
from src.eval.metrics import box_iou, evaluate

SMALL, MEDIUM = 32**2, 96**2
NIGHT_V, DIM_V = 60, 115  # mean HSV V. RDD India test has no night frames (min V=68, backlit day);
# DIM_V ~ 10th percentile separates shadowed/backlit/overcast scenes
CONF_OP = 0.25  # operating confidence for FP/FN counting
COLORS = [(255, 128, 0), (0, 200, 255), (0, 255, 0), (0, 0, 255)]


def load_gt(img: Path, w: int, h: int) -> np.ndarray:
    lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
    rows = []
    for ln in lbl.read_text().splitlines() if lbl.exists() else []:
        c, cx, cy, bw, bh = map(float, ln.split())
        rows.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, float).reshape(-1, 5)


def lighting(img: np.ndarray) -> str:
    v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2].mean()
    return "night" if v < NIGHT_V else "dim" if v < DIM_V else "bright"


def size_of(box: np.ndarray) -> str:
    a = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if a < SMALL else "medium" if a < MEDIUM else "large"


def filter_size(arr: np.ndarray, bucket: str, xyxy: slice) -> np.ndarray:
    return arr[[size_of(b[xyxy]) == bucket for b in arr]].reshape(-1, arr.shape[1])


def summarize(preds: list[np.ndarray], gts: list[np.ndarray]) -> dict:
    r = evaluate(preds, gts, len(NAMES))
    return {
        "images": len(gts),
        "map50": round(r["map50"], 4),
        "map50_95": round(r["map50_95"], 4),
        "per_class": {
            NAMES[c]: {"ap50": round(float(v["ap50"]), 4), "n_gt": v["n_gt"]}
            for c, v in r["per_class"].items()
            if v["n_gt"]
        },
    }


def error_types(pred: np.ndarray, gt: np.ndarray) -> tuple[Counter, Counter, list, list]:
    """At CONF_OP: classify FPs (background / wrong class / localisation / duplicate) and FNs per class."""
    pred = pred[pred[:, 4] >= CONF_OP]
    pred = pred[np.argsort(-pred[:, 4])]
    fp, fn = Counter(), Counter()
    fp_boxes, used = [], np.zeros(len(gt), bool)
    iou = box_iou(pred[:, :4], gt[:, 1:]) if len(gt) else np.zeros((len(pred), 0))
    for i, p in enumerate(pred):
        same = (gt[:, 0] == p[5]) if len(gt) else np.zeros(0, bool)
        ok = np.where((iou[i] >= 0.5) & same & ~used)[0] if len(gt) else []
        if len(ok):
            used[ok[np.argmax(iou[i, ok])]] = True
            continue
        if len(gt) and (iou[i][same] >= 0.5).any():
            kind = "duplicate"
        elif len(gt) and (iou[i][~same] >= 0.5).any():
            kind = "wrong_class"
        elif len(gt) and (iou[i] >= 0.1).any():
            kind = "localisation"
        else:
            kind = "background"
        fp[f"{NAMES[int(p[5])]}:{kind}"] += 1
        fp_boxes.append(p)
    fn_boxes = [g for g, u in zip(gt, used, strict=True) if not u]
    for g in fn_boxes:
        fn[NAMES[int(g[0])]] += 1
    return fp, fn, fp_boxes, fn_boxes


def draw(img: np.ndarray, gt: np.ndarray, fp: list, fn: list, pred: np.ndarray) -> np.ndarray:
    out = img.copy()
    for p in pred[pred[:, 4] >= CONF_OP]:
        x1, y1, x2, y2 = map(int, p[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS[int(p[5])], 2)
        cv2.putText(
            out, f"{NAMES[int(p[5])][:5]} {p[4]:.2f}", (x1, max(12, y1 - 4)), 0, 0.45, COLORS[int(p[5])], 1
        )
    for g in fn:  # missed GT: white dashed-ish thick box
        x1, y1, x2, y2 = map(int, g[1:])
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.putText(out, f"MISSED {NAMES[int(g[0])][:5]}", (x1, y2 + 14), 0, 0.45, (255, 255, 255), 1)
    for p in fp:
        x1, y1, x2, y2 = map(int, p[:4])
        cv2.putText(out, "FP", (x1 + 2, y1 + 14), 0, 0.5, (0, 0, 255), 2)
    return out


def predictor(weights: str, imgsz: int):
    """Returns f(img_bgr) -> (N,6) xyxy,conf,cls. .onnx -> own ORT detector; .pt -> Ultralytics."""
    if weights.endswith(".onnx"):
        from src.inference.detector import Detector

        det = Detector(weights)
        return lambda img: det(img, conf=0.001, iou=0.7)
    from ultralytics import YOLO

    model = YOLO(weights, task="detect")

    def f(img: np.ndarray) -> np.ndarray:
        r = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, verbose=False, max_det=300)[0].boxes
        return np.concatenate(
            [r.xyxy.cpu().numpy(), r.conf.cpu().numpy()[:, None], r.cls.cpu().numpy()[:, None]], 1
        )

    return f


def analyze(weights: str, img_list: Path, name: str, imgsz: int, n_failures: int) -> dict:
    imgs = [Path(p) for p in img_list.read_text().splitlines() if p.strip()]
    manual = {}
    cond_csv = Path("data/conditions.csv")
    if cond_csv.exists():
        with open(cond_csv) as f:
            manual = {r["image"]: r for r in csv.DictReader(f)}
    predict = predictor(weights, imgsz)

    preds, gts, tags, scores = [], [], [], []
    fp_all, fn_all = Counter(), Counter()
    for p in imgs:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = load_gt(p, w, h)
        pred = predict(img)
        preds.append(pred)
        gts.append(gt)
        m = manual.get(p.name, {})
        tags.append(
            {"lighting": m.get("lighting") or lighting(img), "surface": m.get("surface") or "unknown"}
        )
        fp, fn, fpb, fnb = error_types(pred, gt)
        fp_all.update(fp)
        fn_all.update(fn)
        scores.append((len(fpb) + 2 * len(fnb), p, img, gt, fpb, fnb, pred))

    report: dict = {"name": name, "weights": weights, "list": str(img_list), "imgsz": imgsz}
    report["overall"] = summarize(preds, gts)
    for key in ("lighting", "surface"):
        groups = defaultdict(list)
        for i, t in enumerate(tags):
            groups[t[key]].append(i)
        report[f"by_{key}"] = {
            k: summarize([preds[i] for i in ix], [gts[i] for i in ix]) for k, ix in sorted(groups.items())
        }
    report["by_box_size"] = {
        b: summarize(
            [filter_size(p, b, slice(0, 4)) for p in preds], [filter_size(g, b, slice(1, 5)) for g in gts]
        )
        for b in ("small", "medium", "large")
    }
    report[f"false_positives_at_conf_{CONF_OP}"] = dict(fp_all.most_common())
    report[f"false_negatives_at_conf_{CONF_OP}"] = dict(fn_all.most_common())

    out = Path("results/failures") / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    for k, (s, p, img, gt, fpb, fnb, pred) in enumerate(sorted(scores, key=lambda x: -x[0])[:n_failures]):
        if s == 0:
            break
        cv2.imwrite(
            str(out / f"{k:02d}_{p.stem}_fp{len(fpb)}_fn{len(fnb)}.jpg"), draw(img, gt, fpb, fnb, pred)
        )
    Path(f"results/error_analysis_{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "by_surface"}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights")
    ap.add_argument("--list", type=Path, default=Path("data/splits/test.txt"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--failures", type=int, default=15)
    a = ap.parse_args()
    analyze(a.weights, a.list, a.name, a.imgsz, a.failures)

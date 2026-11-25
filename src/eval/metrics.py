"""Minimal detection metrics: box IoU, greedy matching, COCO 101-point AP.

Kept dependency-free so the same code scores PyTorch, ONNX FP32/FP16/INT8 and
any image subset (night, small boxes, in-the-wild) identically.
"""

import numpy as np

IOUS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes a (N,4) and b (M,4) -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """pred (N,6: xyxy, conf, cls), gt (M,5: cls, xyxy) -> tp (N,10) per IoU threshold.

    Greedy by confidence; a GT box is matched at most once per threshold, same class only.
    """
    tp = np.zeros((len(pred), len(IOUS)), bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp
    order = np.argsort(-pred[:, 4])
    iou = box_iou(pred[order, :4], gt[:, 1:])
    same = pred[order, 5][:, None] == gt[:, 0][None, :]
    iou = iou * same
    for t, thr in enumerate(IOUS):
        used = np.zeros(len(gt), bool)
        for i in range(len(order)):
            cand = np.where((iou[i] >= thr) & ~used)[0]
            if len(cand):
                j = cand[np.argmax(iou[i, cand])]
                used[j] = True
                tp[order[i], t] = True
    return tp


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> np.ndarray:
    """AP at each IoU threshold (COCO 101-point interpolation). tp (N,10)."""
    if n_gt == 0:
        return np.full(len(IOUS), np.nan)
    if len(tp) == 0:
        return np.zeros(len(IOUS))
    order = np.argsort(-conf)
    tpc = tp[order].cumsum(0)
    fpc = (~tp[order]).cumsum(0)
    recall = tpc / n_gt
    precision = tpc / (tpc + fpc)
    ap = np.zeros(len(IOUS))
    grid = np.linspace(0, 1, 101)
    for t in range(len(IOUS)):
        p = np.maximum.accumulate(precision[::-1, t])[::-1]  # monotone envelope
        idx = np.searchsorted(recall[:, t], grid, side="left")
        ap[t] = np.mean([p[i] if i < len(p) else 0.0 for i in idx])
    return ap


def evaluate(preds: list[np.ndarray], gts: list[np.ndarray], nc: int) -> dict:
    """Per-class AP50 / AP50-95 and overall mAP over a list of images."""
    tps, confs, clss = [], [], []
    n_gt = np.zeros(nc, int)
    for p, g in zip(preds, gts, strict=True):
        tps.append(match(p, g))
        confs.append(p[:, 4])
        clss.append(p[:, 5])
        for c in g[:, 0].astype(int):
            n_gt[c] += 1
    tp = np.concatenate(tps) if tps else np.zeros((0, 10), bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    cls = np.concatenate(clss) if clss else np.zeros(0)
    per = {}
    for c in range(nc):
        m = cls == c
        ap = average_precision(tp[m], conf[m], int(n_gt[c]))
        per[c] = {"ap50": ap[0], "ap50_95": np.nanmean(ap) if n_gt[c] else np.nan, "n_gt": int(n_gt[c])}
    valid = [v for v in per.values() if v["n_gt"] > 0]
    return {
        "map50": float(np.mean([v["ap50"] for v in valid])) if valid else float("nan"),
        "map50_95": float(np.mean([v["ap50_95"] for v in valid])) if valid else float("nan"),
        "per_class": per,
    }

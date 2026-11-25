"""Torch-free YOLO detector on ONNX Runtime (CPU). Used by benchmark, video and API.

Input: BGR uint8 image. Output: (N, 6) array of x1, y1, x2, y2, conf, cls in
original image pixels.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ["longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"]


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio, pad to size x size with gray 114 (Ultralytics convention)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[top : top + nh, left : left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, (left, top)


def preprocess(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    lb, r, pad = letterbox(img, size)
    x = lb[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0  # BGR->RGB, NCHW
    return np.ascontiguousarray(x), r, pad


def postprocess(
    out: np.ndarray, r: float, pad: tuple[int, int], shape: tuple[int, int], conf: float, iou: float
) -> np.ndarray:
    """out: (1, 4+nc, A) raw YOLOv8 head -> NMS'd boxes in original pixels."""
    p = out[0].T  # (A, 4+nc)
    scores = p[:, 4:]
    cls = scores.argmax(1)
    sc = scores[np.arange(len(p)), cls]
    keep = sc >= conf
    p, cls, sc = p[keep], cls[keep], sc[keep]
    if len(p) == 0:
        return np.zeros((0, 6), np.float32)
    xywh = p[:, :4].copy()
    xywh[:, 0] -= xywh[:, 2] / 2
    xywh[:, 1] -= xywh[:, 3] / 2
    idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), sc.tolist(), cls.tolist(), conf, iou, top_k=300)
    idx = np.array(idx, int).reshape(-1)[:300]
    b = xywh[idx]
    x1 = (b[:, 0] - pad[0]) / r
    y1 = (b[:, 1] - pad[1]) / r
    x2 = x1 + b[:, 2] / r
    y2 = y1 + b[:, 3] / r
    h, w = shape
    boxes = np.stack([x1.clip(0, w), y1.clip(0, h), x2.clip(0, w), y2.clip(0, h)], 1)
    return np.concatenate([boxes, sc[idx, None], cls[idx, None]], 1).astype(np.float32)


class Detector:
    def __init__(self, model: str | Path, threads: int = 0) -> None:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input = inp.name
        self.size = int(inp.shape[2])
        self.fp16 = inp.type == "tensor(float16)"

    def __call__(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.7) -> np.ndarray:
        x, r, pad = preprocess(img, self.size)
        if self.fp16:
            x = x.astype(np.float16)
        out = self.sess.run(None, {self.input: x})[0].astype(np.float32)
        return postprocess(out, r, pad, img.shape[:2], conf, iou)


def draw(img: np.ndarray, dets: np.ndarray) -> np.ndarray:
    colors = [(255, 128, 0), (0, 200, 255), (0, 200, 0), (0, 0, 255)]
    out = img.copy()
    for x1, y1, x2, y2, c, k in dets:
        col = colors[int(k)]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(out, f"{NAMES[int(k)]} {c:.2f}", (int(x1), max(12, int(y1) - 4)), 0, 0.5, col, 2)
    return out

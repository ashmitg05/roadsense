"""Benchmark every model format on the same test images, CPU, batch 1.

Each format runs in its own subprocess so peak RAM is isolated.
Accuracy uses src.eval.metrics for every format (PyTorch included), so the
numbers are directly comparable.

    python -m src.optimize.benchmark --pt runs/<best>/weights/best.pt --models models --tag laptop
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WARMUP = 10


def run_one(model: str, img_list: Path, imgsz: int, threads: int) -> dict:
    import cv2
    import psutil

    from src.eval.analyze import load_gt
    from src.eval.metrics import evaluate

    if model.endswith(".pt"):
        from ultralytics import YOLO

        m = YOLO(model)

        def predict(img: np.ndarray) -> np.ndarray:
            import torch

            torch.set_num_threads(threads or os.cpu_count())
            r = m.predict(img, imgsz=imgsz, conf=0.001, iou=0.7, device="cpu", verbose=False)[0].boxes
            return np.concatenate([r.xyxy.numpy(), r.conf.numpy()[:, None], r.cls.numpy()[:, None]], 1)
    else:
        from src.inference.detector import Detector

        det = Detector(model, threads)

        def predict(img: np.ndarray) -> np.ndarray:
            return det(img, conf=0.001, iou=0.7)

    imgs = [Path(p) for p in img_list.read_text().splitlines() if p.strip()]
    preds, gts, times = [], [], []
    for i, p in enumerate(imgs):
        img = cv2.imread(str(p))
        t = time.perf_counter()
        pred = predict(img)
        if i >= WARMUP:
            times.append((time.perf_counter() - t) * 1000)
        preds.append(pred)
        gts.append(load_gt(p, img.shape[1], img.shape[0]))
    r = evaluate(preds, gts, 4)
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None) or mem.rss  # Windows peak working set; Linux see below
    if sys.platform != "win32":
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    t = np.array(times)
    return {
        "model": model,
        "map50": round(r["map50"], 4),
        "map50_95": round(r["map50_95"], 4),
        "latency_ms_p50": round(float(np.percentile(t, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(t, 95)), 1),
        "fps": round(1000 / float(t.mean()), 1),
        "size_mb": round(Path(model).stat().st_size / 1e6, 1),
        "peak_ram_mb": round(peak / 1e6),
        "images": len(imgs),
    }


def main(a: argparse.Namespace) -> None:
    models = ([str(a.pt)] if a.pt else []) + [str(p) for p in sorted(a.models.glob("model_*.onnx"))]
    rows = []
    for m in models:
        cmd = [
            sys.executable,
            "-m",
            "src.optimize.benchmark",
            "--one",
            m,
            "--list",
            str(a.list),
            "--imgsz",
            str(a.imgsz),
            "--threads",
            str(a.threads),
        ]
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        rows.append(json.loads(out.strip().splitlines()[-1]))
        print(rows[-1])
    base = rows[0]  # first row = reference: PyTorch FP32 if --pt given, else ONNX FP16 (sorted) on the Pi
    for r in rows:
        r["map50_drop_vs_ref"] = round(base["map50"] - r["map50"], 4)
        r["map50_95_drop_vs_ref"] = round(base["map50_95"] - r["map50_95"], 4)
    info = {
        "cpu": platform.processor(),
        "machine": platform.machine(),
        "threads": a.threads or os.cpu_count(),
    }
    Path("results").mkdir(exist_ok=True)
    Path(f"results/benchmark_{a.tag}.json").write_text(json.dumps({"host": info, "rows": rows}, indent=2))
    md = [
        f"CPU: {info['cpu']} ({info['threads']} threads), batch 1, {base['images']} test images, imgsz {a.imgsz}",
        "",
        "| Format | mAP@50 | mAP@50-95 | Δ mAP@50 | p50 ms | p95 ms | FPS | Size MB | Peak RAM MB |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = (
            "PyTorch FP32" if r["model"].endswith(".pt") else Path(r["model"]).stem.replace("model_", "ONNX ")
        )
        md.append(
            f"| {name} | {r['map50']:.3f} | {r['map50_95']:.3f} | {-r['map50_drop_vs_ref']:+.3f} | "
            f"{r['latency_ms_p50']} | {r['latency_ms_p95']} | {r['fps']} | {r['size_mb']} | {r['peak_ram_mb']} |"
        )
    Path(f"results/benchmark_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", type=Path)
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--list", type=Path, default=Path("data/splits/test.txt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--tag", default="laptop")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.one:
        print(json.dumps(run_one(a.one, a.list, a.imgsz, a.threads)))
    else:
        main(a)

"""Train a YOLO model from a config file, evaluate on the test split, save results.

Config (configs/train/*.yaml):
    name: baseline             # run name, results/<name>.json
    model: yolov8n.pt
    data: configs/rdd_india.yaml
    train_list: null           # optional: override the train image list (e.g. oversampled)
    ultralytics: {imgsz: 640, epochs: 50, ...}   # passed straight to model.train()

Every run logs to MLflow (./mlruns) and fixes seed=42, deterministic=True.

    python -m src.train.train configs/train/baseline.yaml
"""

import argparse
import json
import os
from pathlib import Path

import yaml
from ultralytics import YOLO, settings

from src.data.convert import NAMES

SEED = 42


def run_data_yaml(cfg: dict, run_dir: Path) -> str:
    """Write a per-run dataset yaml, optionally swapping the train list."""
    data = yaml.safe_load(Path(cfg["data"]).read_text())
    if cfg.get("train_list"):
        data["train"] = str(Path(cfg["train_list"]).resolve())
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "data.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return str(out)


def road_augmentations() -> list:
    """Dashcam-specific photometric augs: motion blur, rain, fog, low light.

    Pixel-only, so boxes are untouched. Probabilities kept low: each image
    still looks mostly real, and they stack with Ultralytics' mosaic/HSV.
    """
    import albumentations as A

    return [
        A.MotionBlur(blur_limit=(3, 9), p=0.2),
        A.RandomRain(p=0.1),
        A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.1),
        A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.1), p=0.2),  # skew darker: dusk/night
        A.CLAHE(p=0.05),
        A.ImageCompression(quality_range=(40, 90), p=0.2),  # dashcam JPEG artefacts
    ]


def metrics_dict(m) -> dict:
    box = m.box
    per_class = {}
    for j, c in enumerate(box.ap_class_index):
        per_class[NAMES[int(c)]] = {
            "precision": round(float(box.p[j]), 4),
            "recall": round(float(box.r[j]), 4),
            "ap50": round(float(box.ap50[j]), 4),
            "ap50_95": round(float(box.ap[j]), 4),
        }
    return {
        "map50": round(float(box.map50), 4),
        "map50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class": per_class,
        "speed_ms": {k: round(v, 2) for k, v in m.speed.items()},
    }


def train(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    name = cfg["name"]
    settings.update({"mlflow": True})
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "roadsense")
    os.environ.setdefault("MLFLOW_RUN", name)
    run_dir = Path("runs") / name
    data = run_data_yaml(cfg, run_dir)
    args = {
        "seed": SEED,
        "deterministic": True,
        "project": str(Path("runs").resolve()),
        "name": name,
        "exist_ok": True,
        "workers": 0,  # Windows spawn workers each reload CUDA DLLs -> WinError 1455 on 16 GB RAM
    }
    args.update(cfg.get("ultralytics", {}))
    if cfg.get("road_augs"):
        args["augmentations"] = road_augmentations()

    best = run_dir / "weights" / "best.pt"
    csv_path = run_dir / "results.csv"
    epochs_done = len(csv_path.read_text().splitlines()) - 1 if csv_path.exists() else 0
    if best.exists() and epochs_done >= args.get("epochs", 100):
        print(f"{name}: training already complete ({epochs_done} epochs), evaluating only")
    elif (last := run_dir / "weights" / "last.pt").exists() and epochs_done > 0:
        print(f"{name}: resuming from epoch {epochs_done}")  # ckpt stores args incl. augmentations
        YOLO(last).train(resume=True, workers=0)
    else:
        YOLO(cfg["model"]).train(data=data, **args)
    # Ultralytics logs cumulative training seconds per epoch; last row = total train time
    train_min = float(csv_path.read_text().splitlines()[-1].split(",")[1]) / 60

    imgsz = args.get("imgsz", 640)
    # plots=False: the end-of-run plot rendering hit MemoryError on this 16 GB laptop;
    # the confusion matrix is saved as numbers in the JSON instead
    test = YOLO(best).val(
        data=data,
        split="test",
        imgsz=imgsz,
        plots=False,
        workers=0,  # val() does not inherit train workers; default 8 OOMs here
        project=str(Path("runs").resolve()),
        name=f"{name}_test",
        exist_ok=True,
    )
    result = {
        "name": name,
        "config": cfg,
        "seed": SEED,
        "weights": str(best),
        "train_minutes": round(train_min, 1),
        "test": metrics_dict(test),
        # rows = predicted, cols = true; classes + background (Ultralytics convention)
        "confusion_matrix": test.confusion_matrix.matrix.astype(int).tolist(),
    }
    Path("results").mkdir(exist_ok=True)
    (Path("results") / f"{name}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["test"], indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    train(ap.parse_args().config)

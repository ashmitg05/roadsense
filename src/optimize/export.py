"""Export best.pt -> ONNX FP32, FP16 and INT8 (static, QDQ, ~300 calibration images).

Two INT8 variants:
  int8       every Conv quantised
  int8_head  Detect head (model.22) left in FP32: the box/class branches
             are the most sensitive to quantisation error, and this costs
             little speed because the backbone dominates compute.

    python -m src.optimize.export runs/<best>/weights/best.pt --imgsz 640
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from src.inference.detector import preprocess


class Calib(CalibrationDataReader):
    def __init__(self, imgs: list[Path], size: int, name: str) -> None:
        self.it = iter([{name: preprocess(cv2.imread(str(p)), size)[0]} for p in imgs])

    def get_next(self) -> dict | None:
        return next(self.it, None)


def export(weights: Path, out: Path, imgsz: int, calib_list: Path, n_calib: int, seed: int) -> None:
    from ultralytics import YOLO

    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model_fp32.onnx"
    src = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
    shutil.move(src, fp32)

    from onnxconverter_common import float16

    m = onnx.load(fp32)
    del m.graph.value_info[:]  # stale fp32 shape info breaks the converter's Resize casts
    m16 = float16.convert_float_to_float16(m, keep_io_types=False)
    onnx.save(m16, out / "model_fp16.onnx")

    pre = out / "_pre.onnx"
    quant_pre_process(str(fp32), str(pre))
    imgs = [Path(p) for p in calib_list.read_text().splitlines() if p.strip()]
    imgs = sorted(set(imgs))
    calib = random.Random(seed).sample(imgs, min(n_calib, len(imgs)))
    inp = onnx.load(pre).graph.input[0].name
    head = [n.name for n in onnx.load(pre).graph.node if "/model.22/" in n.name]
    for tag, exclude in (("int8", []), ("int8_head", head)):
        quantize_static(
            str(pre),
            str(out / f"model_{tag}.onnx"),
            Calib(calib, imgsz, inp),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
            nodes_to_exclude=exclude,
        )
    pre.unlink()
    for f in sorted(out.glob("model_*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-list", type=Path, default=Path("data/splits/train.txt"))
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    export(a.weights, a.out, a.imgsz, a.calib_list, a.n_calib, a.seed)

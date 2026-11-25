"""API + inference pipeline tests. Need the committed INT8 model (models/model_int8_head.onnx)."""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

MODEL = Path(os.environ.get("TEST_MODEL", "models/model_int8_head.onnx"))
FIXTURE = Path("tests/fixtures/road.jpg")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="model not exported yet")


@pytest.fixture(scope="module")
def client():
    os.environ["MODEL_PATH"] = str(MODEL)
    os.environ["RATE_LIMIT_PER_MIN"] = "1000"
    from fastapi.testclient import TestClient

    from src.inference.api import app

    return TestClient(app)


def test_detector_output_shape_and_bounds() -> None:
    from src.inference.detector import Detector

    img = cv2.imread(str(FIXTURE))
    d = Detector(MODEL)(img, conf=0.05)
    assert d.ndim == 2 and d.shape[1] == 6
    h, w = img.shape[:2]
    assert (d[:, [0, 2]] <= w).all() and (d[:, [1, 3]] <= h).all() and (d[:, :2] <= d[:, 2:4]).all()
    assert set(d[:, 5].astype(int)) <= {0, 1, 2, 3}


def test_health(client) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_detect_image(client) -> None:
    r = client.post("/detect/image", files={"file": ("road.jpg", FIXTURE.read_bytes(), "image/jpeg")})
    assert r.status_code == 200
    body = r.json()
    assert set(body["counts"]) == {"longitudinal_crack", "transverse_crack", "alligator_crack", "pothole"}
    assert body["inference_ms"] > 0 and body["image_jpeg_base64"]


def test_rejects_bad_uploads(client) -> None:
    assert client.post("/detect/image", files={"file": ("x.txt", b"hello", "text/plain")}).status_code == 415
    assert (
        client.post("/detect/image", files={"file": ("x.jpg", b"notjpeg", "image/jpeg")}).status_code == 400
    )
    assert (
        client.post("/detect/video", files={"file": ("x.exe", b"MZ", "application/octet-stream")}).status_code
        == 415
    )


def make_video(path: Path, seconds: float, fps: int = 5) -> None:
    img = cv2.resize(cv2.imread(str(FIXTURE)), (320, 320))
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 320))
    for i in range(int(seconds * fps)):
        w.write(np.roll(img, i, axis=0))
    w.release()


def test_video_job_roundtrip(client, tmp_path: Path) -> None:
    v = tmp_path / "clip.mp4"
    make_video(v, 2)
    r = client.post("/detect/video", files={"file": ("clip.mp4", v.read_bytes(), "video/mp4")})
    assert r.status_code == 200
    jid = r.json()["job_id"]
    for _ in range(120):
        s = client.get(f"/jobs/{jid}").json()
        if s["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert s["status"] == "done", s
    assert s["result"]["frames"] == 10
    assert client.get(f"/jobs/{jid}/video").status_code == 200


def test_video_too_long_rejected(client, tmp_path: Path) -> None:
    v = tmp_path / "long.mp4"
    make_video(v, 32, fps=2)
    r = client.post("/detect/video", files={"file": ("long.mp4", v.read_bytes(), "video/mp4")})
    assert r.status_code == 413

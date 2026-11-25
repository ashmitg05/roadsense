"""FastAPI service: INT8 ONNX detector on CPU.

GET  /health
POST /detect/image          multipart file -> boxes, counts, annotated JPEG (base64), ms
POST /detect/video          multipart file (<=50 MB, <=30 s) -> {job_id}
GET  /jobs/{job_id}         status + result
GET  /jobs/{job_id}/video   annotated .webm

MODEL_PATH=models/model_int8_head.onnx uvicorn src.inference.api:app
"""

import base64
import os
import tempfile
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.inference.detector import NAMES, Detector, draw
from src.inference.video import process_video

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model_int8_head.onnx")
MAX_IMAGE_MB, MAX_VIDEO_MB, MAX_VIDEO_S = 10, 50, 30
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))
JOB_TTL_S = 3600
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

app = FastAPI(title="RoadSense")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
detector = Detector(MODEL_PATH)
# ponytail: jobs live in process memory and one worker thread; on Cloud Run set
# max-instances=1 (or move jobs to GCS + Cloud Tasks) if videos must survive scale-out
executor = ThreadPoolExecutor(max_workers=1)
jobs: dict[str, dict] = {}
hits: dict[str, deque] = defaultdict(deque)
lock = threading.Lock()


def client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")  # Cloud Run sets this; first hop is the client
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")


def rate_limit(req: Request) -> None:
    now, ip = time.monotonic(), client_ip(req)
    with lock:
        q = hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            raise HTTPException(429, f"rate limit: {RATE_LIMIT} requests/minute")
        q.append(now)


async def read_capped(f: UploadFile, max_mb: int) -> bytes:
    data = await f.read(max_mb * 2**20 + 1)
    if len(data) > max_mb * 2**20:
        raise HTTPException(413, f"file larger than {max_mb} MB")
    if not data:
        raise HTTPException(400, "empty file")
    return data


def to_json(dets: np.ndarray) -> list[dict]:
    return [
        {
            "box": [round(float(v), 1) for v in d[:4]],
            "confidence": round(float(d[4]), 3),
            "class": NAMES[int(d[5])],
        }
        for d in dets
    ]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": Path(MODEL_PATH).name}


@app.post("/detect/image")
async def detect_image(req: Request, file: UploadFile) -> dict:
    rate_limit(req)
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(415, f"unsupported type {file.content_type}; use jpeg/png/webp")
    data = await read_capped(file, MAX_IMAGE_MB)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not decode image")
    if max(img.shape[:2]) > 1920:  # bound work per request
        s = 1920 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    t = time.perf_counter()
    dets = detector(img)
    ms = (time.perf_counter() - t) * 1000
    ok, jpg = cv2.imencode(".jpg", draw(img, dets), [cv2.IMWRITE_JPEG_QUALITY, 85])
    counts = Counter(NAMES[int(k)] for k in dets[:, 5])
    return {
        "detections": to_json(dets),
        "counts": {n: counts.get(n, 0) for n in NAMES},
        "inference_ms": round(ms, 1),
        "image_jpeg_base64": base64.b64encode(jpg.tobytes()).decode(),
    }


def run_job(job_id: str, src: Path) -> None:
    job = jobs[job_id]
    job["status"] = "running"
    try:
        out = src.with_name(f"{job_id}_out.webm")
        res = process_video(src, detector, out, every=2, max_seconds=MAX_VIDEO_S)
        res.pop("video")
        job.update(status="done", result=res, output=str(out))
    except Exception as e:  # noqa: BLE001 - surface any failure to the client
        job.update(status="error", error=str(e))
    finally:
        src.unlink(missing_ok=True)


def purge_old_jobs() -> None:
    now = time.time()
    for jid in [j for j, v in jobs.items() if now - v["created"] > JOB_TTL_S]:
        job = jobs.pop(jid)
        if job.get("output"):
            Path(job["output"]).unlink(missing_ok=True)


@app.post("/detect/video")
async def detect_video(req: Request, file: UploadFile) -> dict:
    rate_limit(req)
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXT:
        raise HTTPException(415, f"unsupported video extension '{ext}'")
    data = await read_capped(file, MAX_VIDEO_MB)
    job_id = uuid.uuid4().hex
    src = Path(tempfile.gettempdir()) / f"{job_id}{ext}"
    src.write_bytes(data)
    cap = cv2.VideoCapture(str(src))
    fps, n = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    readable = cap.isOpened() and cap.read()[0]
    cap.release()
    if not readable or not fps:
        src.unlink(missing_ok=True)
        raise HTTPException(400, "could not decode video")
    duration = n / fps
    if duration > MAX_VIDEO_S + 0.5:
        src.unlink(missing_ok=True)
        raise HTTPException(413, f"video is {duration:.1f}s; max {MAX_VIDEO_S}s")
    purge_old_jobs()
    jobs[job_id] = {"status": "queued", "created": time.time()}
    executor.submit(run_job, job_id, src)
    return {"job_id": job_id, "duration_s": round(duration, 1)}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job")
    return {k: v for k, v in job.items() if k not in ("output", "created")}


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "video not ready")
    return FileResponse(job["output"], media_type="video/webm")

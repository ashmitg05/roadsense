"""Video pipeline: detect (ONNX) -> ByteTrack -> annotate -> unique damage counts.

ByteTrack is class-agnostic, so each track's class is the majority vote of its
per-frame classes. A track counts once, and only if it was matched in at least
`min_frames` processed frames (filters one-frame flickers).

    python -m src.inference.video in.mp4 --model models/model.onnx --every 2 --out out.mp4
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from src.inference.detector import NAMES, Detector, draw


def process_video(
    src: Path,
    det: Detector,
    out: Path | None = None,
    every: int = 1,
    conf: float = 0.1,
    min_frames: int = 3,
    max_seconds: float | None = None,
) -> dict:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker = ByteTrackTracker(
        frame_rate=fps / every,
        lost_track_buffer=int(fps / every),  # keep lost tracks ~1 s
        track_activation_threshold=0.4,
        high_conf_det_threshold=0.4,
        minimum_consecutive_frames=2,
    )
    # .webm/VP8 plays in browsers with stock opencv wheels; mp4v is for local viewing
    fourcc = cv2.VideoWriter_fourcc(*("VP80" if out and out.suffix == ".webm" else "mp4v"))
    writer = cv2.VideoWriter(str(out), fourcc, fps / every, (w, h)) if out else None

    votes: dict[int, Counter] = defaultdict(Counter)
    first_seen: dict[int, float] = {}
    n_frames = n_proc = 0
    infer_s = 0.0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (max_seconds and n_frames / fps >= max_seconds):
            break
        n_frames += 1
        if (n_frames - 1) % every:
            continue
        n_proc += 1
        t = time.perf_counter()
        d = det(frame, conf=conf)
        infer_s += time.perf_counter() - t
        dets = sv.Detections(xyxy=d[:, :4], confidence=d[:, 4], class_id=d[:, 5].astype(int))
        tracked = tracker.update(dets)
        shown = []
        for box, c, k, tid in zip(
            tracked.xyxy, tracked.confidence, tracked.class_id, tracked.tracker_id, strict=True
        ):
            if tid is None or tid < 0:
                continue
            votes[int(tid)][int(k)] += 1
            first_seen.setdefault(int(tid), n_frames / fps)
            shown.append([*box, c, k])
        if writer:
            vis = draw(frame, np.array(shown).reshape(-1, 6))
            cv2.putText(
                vis,
                f"unique so far: {sum(sum(v.values()) >= min_frames for v in votes.values())}",
                (10, 30),
                0,
                0.8,
                (255, 255, 255),
                2,
            )
            writer.write(vis)
    cap.release()
    if writer:
        writer.release()
    wall = time.perf_counter() - t0

    counts = Counter()
    tracks = []
    for tid, v in votes.items():
        if sum(v.values()) >= min_frames:
            cls = v.most_common(1)[0][0]
            counts[NAMES[cls]] += 1
            tracks.append({"id": tid, "class": NAMES[cls], "t": round(first_seen[tid], 2)})
    return {
        "video": str(src),
        "frames": n_frames,
        "processed_frames": n_proc,
        "every_nth": every,
        "video_fps": round(fps, 2),
        "unique_counts": {n: counts.get(n, 0) for n in NAMES},
        "tracks": tracks,
        "inference_fps": round(n_proc / infer_s, 1) if infer_s else 0.0,
        "pipeline_fps_in_video_frames": round(n_frames / wall, 1) if wall else 0.0,
    }


def gps_map(tracks: list[dict], gpx: Path, out: Path) -> None:
    """Place each track at the GPX point nearest its first-seen time (GPX start = video start)."""
    import folium
    import gpxpy

    pts = [p for t in gpxpy.parse(gpx.read_text()).tracks for s in t.segments for p in s.points]
    t0 = pts[0].time
    secs = np.array([(p.time - t0).total_seconds() for p in pts])
    m = folium.Map(location=[pts[0].latitude, pts[0].longitude], zoom_start=16)
    folium.PolyLine([(p.latitude, p.longitude) for p in pts], weight=3).add_to(m)
    colors = {
        "pothole": "red",
        "alligator_crack": "orange",
        "longitudinal_crack": "blue",
        "transverse_crack": "purple",
    }
    for tr in tracks:
        p = pts[int(np.abs(secs - tr["t"]).argmin())]
        folium.CircleMarker(
            (p.latitude, p.longitude),
            radius=6,
            color=colors[tr["class"]],
            fill=True,
            popup=f"{tr['class']} #{tr['id']} @ {tr['t']}s",
        ).add_to(m)
    m.save(str(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--model", default="models/model.onnx")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--every", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--gpx", type=Path, help="optional GPX track -> damage map html")
    a = ap.parse_args()
    res = process_video(a.video, Detector(a.model), a.out, a.every)
    if a.gpx:
        gps_map(res["tracks"], a.gpx, (a.out or a.video).with_suffix(".map.html"))
    print(json.dumps({k: v for k, v in res.items() if k != "tracks"}, indent=2))

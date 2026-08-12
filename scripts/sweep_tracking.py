"""Diagnose ID fragmentation by sweeping tracker parameters.

Detection is the slow part, so it runs once and is cached to disk. The
tracker is then re-run over the cached boxes for each parameter set, which
takes a second rather than four minutes.

If the unique-person count collapses as `max_missing` grows, the tracker is
splitting single people into several tracks — which means the headline
number is measuring the tracker's weaknesses, not the footage.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humandetect.analytics import build_report  # noqa: E402
from humandetect.detector import Detection, PersonDetector  # noqa: E402
from humandetect.tracking import PersonTracker  # noqa: E402

VIDEO = Path("samples/Samplevideo2.mp4")
CACHE = Path("outputs/detections.pkl")


def cache_detections() -> dict:
    if CACHE.exists():
        with CACHE.open("rb") as fh:
            return pickle.load(fh)

    detector = PersonDetector()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame: list[list[Detection]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        per_frame.append(detector.detect(frame))
        if len(per_frame) % 500 == 0:
            print(f"  detecting {len(per_frame)}…", flush=True)
    cap.release()

    payload = {"fps": fps, "width": width, "height": height, "frames": per_frame}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(payload, fh)
    return payload


def run(payload: dict, max_missing: int, min_frames: int) -> dict:
    diagonal = (payload["width"] ** 2 + payload["height"] ** 2) ** 0.5
    tracker = PersonTracker(frame_diagonal=diagonal, max_missing=max_missing)
    for i, dets in enumerate(payload["frames"]):
        tracker.update(dets, i)

    all_tracks = list(tracker.finish())
    people = [t for t in all_tracks if t.frames_seen >= min_frames]
    report = build_report(people, all_tracks, len(payload["frames"]), payload["fps"])
    return {
        "max_missing": max_missing,
        "min_frames": min_frames,
        "unique": report.unique_people,
        "mean_dwell": round(report.mean_dwell, 2),
        "median_dwell": round(report.median_dwell, 2),
        "peak": report.peak_concurrent,
    }


def main() -> int:
    payload = cache_detections()
    fps = payload["fps"]
    print(f"\nCached {len(payload['frames']):,} frames @ {fps:.0f}fps\n")

    print(f"  {'max_missing':>11} {'min_frames':>10} {'unique':>7} {'mean_dwell':>11} {'median':>7} {'peak':>5}")
    print(f"  {'-' * 11} {'-' * 10} {'-' * 7} {'-' * 11} {'-' * 7} {'-' * 5}")

    rows = []
    for max_missing in (5, 15, 30, 60, 90):
        for min_frames in (5, 15, 30):
            r = run(payload, max_missing, min_frames)
            rows.append(r)
            print(
                f"  {r['max_missing']:>11} {r['min_frames']:>10} {r['unique']:>7} "
                f"{r['mean_dwell']:>10.2f}s {r['median_dwell']:>6.2f}s {r['peak']:>5}"
            )

    Path("outputs/sweep.json").write_text(json.dumps(rows, indent=2))
    print("\nWrote outputs/sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

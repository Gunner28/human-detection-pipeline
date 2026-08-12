"""Phase 1 verification: run the detector across the full sample video.

Measures the same footage twice — once counting raw model output the way
the original notebook did, once after duplicate suppression — so the
difference between the two is a number rather than an assertion.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humandetect.detector import PersonDetector, draw  # noqa: E402

VIDEO = Path("samples/Samplevideo2.mp4")
OUT_VIDEO = Path("outputs/Samplevideo2_detected.mp4")
CONF = 0.5


def main() -> int:
    detector = PersonDetector()
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"Cannot open {VIDEO}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    frames = 0
    raw_total = 0
    dedup_total = 0
    frames_with_person = 0
    max_in_frame = 0
    started = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        raw = detector.detect(frame, confidence_threshold=CONF, deduplicate=False)
        people = detector.detect(frame, confidence_threshold=CONF, deduplicate=True)

        raw_total += len(raw)
        dedup_total += len(people)
        if people:
            frames_with_person += 1
            max_in_frame = max(max_in_frame, len(people))

        writer.write(draw(frame, people))

        if frames % 500 == 0:
            print(f"  {frames}/{total} frames…", flush=True)

    elapsed = time.time() - started
    cap.release()
    writer.release()

    print(f"\nVideo:               {VIDEO.name}  {width}x{height}  {fps:.0f}fps")
    print(f"Frames processed:    {frames:,} ({frames / fps / 60:.1f} min)")
    print(f"Wall time:           {elapsed:.0f}s  ({frames / elapsed:.1f} fps throughput)")
    print()
    print(f"Person boxes, raw:   {raw_total:,}   <- what the notebook counted")
    print(f"Person boxes, dedup: {dedup_total:,}")
    inflation = (raw_total / dedup_total) if dedup_total else 0
    print(f"Inflation factor:    {inflation:.2f}x")
    print()
    print(f"Frames with >=1 person: {frames_with_person:,} ({frames_with_person / frames:.1%})")
    print(f"Most people in a frame: {max_in_frame}")
    print(f"Mean people per frame:  {dedup_total / frames:.2f}")
    print(f"\nWrote {OUT_VIDEO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

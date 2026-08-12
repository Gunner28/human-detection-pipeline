"""Benchmark detector backends against the labelled frames.

Same frames, same ground truth, same size filter, same suppression. The
only variable is the model, so any difference in the numbers is
attributable to it.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humandetect.backends import load_backend  # noqa: E402
from humandetect.evaluate import CountReport  # noqa: E402

GT_PATH = Path("outputs/ground_truth.json")
BACKENDS = ["ssd", "yolo"]
CONFIDENCES = [0.3, 0.4, 0.5, 0.6]


def main() -> int:
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))
    min_height = gt["min_height"]
    frames = {int(k): v for k, v in gt["frames"].items()}

    cap = cv2.VideoCapture(gt["video"])
    if not cap.isOpened():
        print(f"Cannot open {gt['video']}")
        return 1

    # Read the labelled frames once so decoding cost is not attributed to
    # either model.
    images: dict[int, cv2.typing.MatLike] = {}
    for idx in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            images[idx] = frame
    cap.release()

    actual_total = sum(frames.values())
    print(f"Ground truth: {len(frames)} frames, {actual_total} people >= {min_height}px\n")

    rows = []
    for backend_name in BACKENDS:
        try:
            backend = load_backend(backend_name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {backend_name}: unavailable ({exc})")
            continue

        # Warm up: first inference includes model load and lazy init.
        first = next(iter(images.values()))
        backend.detect(first, confidence_threshold=0.5)

        print(f"{backend.name}")
        print(f"  {'conf':>5} {'detected':>9} {'ratio':>7} {'MAE':>6} {'bias':>7} {'ms/frame':>9}  per-frame")
        for conf in CONFIDENCES:
            report = CountReport(min_height=min_height, confidence=conf)
            started = time.perf_counter()
            for idx, actual in sorted(frames.items()):
                dets = [
                    d
                    for d in backend.detect(images[idx], confidence_threshold=conf)
                    if d.box[3] >= min_height
                ]
                report.frame_indices.append(idx)
                report.actual.append(actual)
                report.predicted.append(len(dets))
            elapsed_ms = (time.perf_counter() - started) * 1000 / max(len(frames), 1)

            print(
                f"  {conf:>5} {report.total_predicted:>9} {report.ratio:>6.2f}x "
                f"{report.mean_absolute_error:>6.2f} {report.bias:>+7.2f} {elapsed_ms:>9.1f}  {report.predicted}"
            )
            rows.append(
                {"backend": backend.name, "ms_per_frame": round(elapsed_ms, 1), **report.to_dict()}
            )
        print()

    Path("outputs/backend_comparison.json").write_text(json.dumps(rows, indent=2))
    print("Wrote outputs/backend_comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

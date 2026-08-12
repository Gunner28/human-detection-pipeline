"""Find shot boundaries in the sample footage.

Motivation: the sampled evaluation frames show wildly different framing —
a blurred close-up, a wide high street, a ground-level shot of legs. That
suggests the clip is an edited montage rather than one continuous take.

It matters because tracking assumes continuity. Across a hard cut, every
person in the new shot is unmatched, so each cut manufactures a fresh set
of "unique people". If the clip has many cuts, the headline count is
measuring the edit, not the crowd.

Detection method: mean absolute difference between consecutive frames on a
downscaled grayscale image. A hard cut produces a spike far above the
rolling baseline of ordinary motion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("samples/Samplevideo2.mp4")
THRESHOLD_MULTIPLIER = 4.0  # spike must exceed this * median difference


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"Cannot open {VIDEO}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    diffs: list[float] = []
    prev = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diffs.append(float(np.mean(cv2.absdiff(small, prev))))
        prev = small
    cap.release()

    if not diffs:
        print("No frames read.")
        return 1

    arr = np.array(diffs)
    baseline = float(np.median(arr))
    threshold = baseline * THRESHOLD_MULTIPLIER

    cuts = [i + 1 for i, d in enumerate(arr) if d > threshold]

    # Collapse cuts within a few frames of each other (a dissolve registers
    # as several consecutive spikes, but it is still one boundary).
    collapsed: list[int] = []
    for c in cuts:
        if not collapsed or c - collapsed[-1] > 5:
            collapsed.append(c)

    print(f"Frames analysed:   {len(arr) + 1:,}")
    print(f"Median frame diff: {baseline:.2f}")
    print(f"Cut threshold:     {threshold:.2f}")
    print(f"\nShot boundaries detected: {len(collapsed)}")
    for c in collapsed:
        print(f"  frame {c:>5}  t={c / fps:6.1f}s")

    if collapsed:
        bounds = [0] + collapsed + [len(arr) + 1]
        lengths = [(bounds[i + 1] - bounds[i]) / fps for i in range(len(bounds) - 1)]
        print(f"\nShots: {len(lengths)}")
        print(f"Mean shot length:   {np.mean(lengths):.1f}s")
        print(f"Median shot length: {np.median(lengths):.1f}s")
        print(f"Shortest / longest: {min(lengths):.1f}s / {max(lengths):.1f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

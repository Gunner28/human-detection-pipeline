"""Extract a stratified sample of frames for ground-truth labelling.

Frames are taken at even intervals across the whole clip so the sample
spans quiet and busy stretches alike. Labelling a contiguous block would
bias the result toward whatever happens to be in that block.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

VIDEO = Path("samples/Samplevideo2.mp4")
OUT_DIR = Path("outputs/eval_frames")
SAMPLE_COUNT = 12


def main() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        print(f"Cannot open {VIDEO}")
        return 1

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    step = total // (SAMPLE_COUNT + 1)
    indices = [step * (i + 1) for i in range(SAMPLE_COUNT)]

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        # Upscale 2x: the frames are 854x480 and distant pedestrians are
        # only a few pixels tall, which makes careful counting hard.
        big = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        path = OUT_DIR / f"frame_{idx:05d}.png"
        cv2.imwrite(str(path), big)
        print(f"  {path.name}  (t={idx / fps:.1f}s)")

    cap.release()
    print(f"\nWrote {len(indices)} frames to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

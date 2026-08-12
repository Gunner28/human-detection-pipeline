"""Check that the local MOT17 copy is complete and internally consistent.

The download is 5.6 GB and is not in git, so nothing else in this
repository can prove it arrived intact. This does, and fails loudly if it
did not. Logic lives in humandetect/mot17.py; this is the command line
around it.

    python scripts/verify_mot17.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humandetect import mot17  # noqa: E402


def main() -> int:
    try:
        stats = mot17.verify()
    except mot17.MOT17Error as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    train = stats["train"]
    # One row per scene, not per folder: the three detector copies are the
    # same video, so listing all 21 would triple-count.
    scenes = [s for s in train if s.name.endswith(f"-{mot17.CANONICAL_VARIANT}")]

    print(f"MOT17 verified at {mot17.MOT_ROOT}\n")
    print(f"{'sequence':<18}{'frames':>8}{'resolution':>12}{'fps':>5}"
          f"{'gt rows':>10}{'evaluated':>11}{'ids':>6}")
    for s in scenes:
        label = s.name.replace(f"-{mot17.CANONICAL_VARIANT}", "")
        print(f"{label:<18}{s.length:>8}{s.resolution:>12}{s.frame_rate:>5}"
              f"{s.gt_rows:>10}{s.gt_evaluated:>11}{s.gt_ids:>6}")

    frames = sum(s.length for s in scenes)
    rows = sum(s.gt_rows for s in scenes)
    evaluated = sum(s.gt_evaluated for s in scenes)
    ids = sum(s.gt_ids for s in scenes)
    print(f"{'TOTAL (7 scenes)':<18}{frames:>8}{'':>12}{'':>5}"
          f"{rows:>10}{evaluated:>11}{ids:>6}")

    dropped = rows - evaluated
    print(
        f"\n{dropped} of {rows} train gt rows ({100 * dropped / rows:.1f}%) are not "
        f"evaluated — conf flag 0, or a class other than "
        f"{mot17.EVALUATED_CLASS} (pedestrian)."
    )
    combined: Counter = Counter()
    for s in scenes:
        combined.update(s.class_counts)
    for class_id, count in sorted(combined.items(), key=lambda kv: -kv[1]):
        print(f"    class {class_id:>2} {mot17.CLASS_NAMES[class_id]:<24}{count:>9}")

    test_frames = sum(
        s.length for s in stats["test"]
        if s.name.endswith(f"-{mot17.CANONICAL_VARIANT}")
    )
    print(
        f"\ntest split: 7 scenes x 3 detectors, {test_frames} frames, no ground "
        f"truth (scored by the MOTChallenge server only)."
    )
    print(
        "\nOK  42 sequence folders = 14 scenes x 3 detectors. Score on the 7 "
        "train scenes; averaging all 21 train folders triple-counts them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

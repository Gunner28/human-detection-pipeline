"""Run the pipeline on MOT17 and score it — one command, one number.

Detection and scoring stay separate in humandetect/benchmark.py, so a
scoring change never forces a re-detection. This runs them back to back
because that is what you almost always want; use --score-only to re-score
results that already exist.

    python scripts/benchmark_mot17.py                       # all 7 scenes, yolo
    python scripts/benchmark_mot17.py --scenes MOT17-02     # one scene
    python scripts/benchmark_mot17.py --backend ssd --name ssd-baseline
    python scripts/benchmark_mot17.py --score-only --name ssd-baseline

Setup once:  ./scripts/setup_trackeval.sh
Dataset:     MOT17.zip from motchallenge.net, unzipped to ./MOT17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from humandetect import mot17  # noqa: E402
from humandetect.benchmark import (  # noqa: E402
    COUNT_FIELDS,
    BenchmarkConfig,
    evaluate,
    run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="scene names, e.g. MOT17-02 (default: all 7 train scenes)",
    )
    parser.add_argument(
        "--backend",
        default="yolo",
        help="ssd | yolo | yolo:<weights>.pt (default: yolo)",
    )
    parser.add_argument("--imgsz", type=int, default=1280,
                        help="max YOLO inference size; capped at native per sequence")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--compensate", action="store_true",
                        help="camera-motion compensation (MOT17-05/10/11/13 move)")
    parser.add_argument("--no-cap-imgsz", action="store_true",
                        help="allow upscaling past a sequence's native resolution")
    parser.add_argument("--name", default=None,
                        help="run name; results go to outputs/mot17/<name>/")
    parser.add_argument("--score-only", action="store_true",
                        help="skip detection, score existing results")
    args = parser.parse_args()

    scenes = args.scenes or [f"MOT17-{s}" for s in mot17.TRAIN_SCENES]
    name = args.name or f"{args.backend.replace(':', '-')}-{args.imgsz}-c{args.confidence}"

    config = BenchmarkConfig(
        backend=args.backend,
        imgsz=args.imgsz,
        confidence=args.confidence,
        min_frames=args.min_frames,
        compensate=args.compensate,
        cap_imgsz_to_native=not args.no_cap_imgsz,
    )

    try:
        if not args.score_only:
            stats = run(scenes, name, config)
            total_frames = sum(s.frames for s in stats)
            total_dets = sum(s.detections for s in stats)
            total_seconds = sum(s.seconds for s in stats)
            print(
                f"\n{len(stats)} sequence(s), {total_frames} frames, "
                f"{total_dets} detections, {total_frames / total_seconds:.1f} fps mean"
            )

        result = evaluate(name, scenes if args.score_only else None)
    except (mot17.MOT17Error, RuntimeError) as exc:
        print(f"FAIL  {exc}", file=sys.stderr)
        return 1

    print(f"\nrun: {name}    {config.describe()}\n")
    print(result.table())

    print(f"\n{'':<20}" + "".join(f"{n:>10}" for _f, n in COUNT_FIELDS))
    for seq in result.sequences:
        counts = result.counts(seq)
        print(f"{seq:<20}" + "".join(f"{counts[n]:>10}" for _f, n in COUNT_FIELDS))
    combined = result.counts()
    print(f"{'COMBINED':<20}" + "".join(f"{combined[n]:>10}" for _f, n in COUNT_FIELDS))

    print(f"\n{result.summary()}")
    print(
        "Private detections (this repo's detector), not the public det/det.txt — "
        "comparable only with other private-detection results."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

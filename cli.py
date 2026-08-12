"""Human detection pipeline — command line.

    python cli.py image samples/manbenz.png
    python cli.py video samples/Samplevideo2.mp4 -o outputs/annotated.mp4
    python cli.py webcam
    python cli.py analyse samples/Samplevideo2.mp4        # audience metrics
    python cli.py image samples/manbenz.png --all-classes --raw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from humandetect.analytics import build_report
from humandetect.counting import CountingLine, CrossingCounter
from humandetect.detector import PersonDetector, draw
from humandetect.motion import CameraMotionEstimator
from humandetect.tracking import PersonTracker


def cmd_image(args: argparse.Namespace) -> int:
    frame = cv2.imread(args.path)
    if frame is None:
        print(f"Cannot read image: {args.path}")
        return 1

    detector = PersonDetector()
    dets = detector.detect(
        frame,
        confidence_threshold=args.confidence,
        people_only=not args.all_classes,
        deduplicate=not args.raw,
    )

    print(f"{args.path}: {len(dets)} detection(s)")
    for d in dets:
        print(f"  {d.label:<14} {d.confidence:.3f}  box={d.box}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, draw(frame, dets))
        print(f"Wrote {args.output}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Cannot open video: {args.path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    detector = PersonDetector()
    frames = detections = frames_with_person = 0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames += 1

            dets = detector.detect(
                frame,
                confidence_threshold=args.confidence,
                people_only=not args.all_classes,
            )
            detections += len(dets)
            if dets:
                frames_with_person += 1

            if writer is not None:
                writer.write(draw(frame, dets))
            if args.show:
                cv2.imshow("Human detection", draw(frame, dets))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frames % 500 == 0:
                print(f"  {frames}/{total} frames…", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = max(time.time() - started, 1e-6)
    print(f"\nFrames:            {frames:,}")
    print(f"Throughput:        {frames / elapsed:.1f} fps")
    print(f"Detections:        {detections:,}")
    print(f"Frames with a hit: {frames_with_person:,} ({frames_with_person / max(frames, 1):.1%})")
    print(f"Mean per frame:    {detections / max(frames, 1):.2f}")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def cmd_webcam(args: argparse.Namespace) -> int:
    """Live detection. The original notebook called `cv2.videoCapture(1)` —
    lowercase `v`, which is not a function, and index 1 rather than the
    default camera at 0."""
    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened() and args.device != 0:
        print(f"Camera {args.device} unavailable, falling back to 0…")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open any camera.")
        return 1

    detector = PersonDetector()
    print("Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = detector.detect(
                frame,
                confidence_threshold=args.confidence,
                people_only=not args.all_classes,
            )
            cv2.imshow("Human detection — press q to quit", draw(frame, dets))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Detect + track across a video, then report audience metrics."""
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Cannot open video: {args.path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diagonal = (width**2 + height**2) ** 0.5

    detector = PersonDetector()
    tracker = PersonTracker(frame_diagonal=diagonal)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    frames = 0
    started = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = detector.detect(frame, confidence_threshold=args.confidence)
            pairs = tracker.update(dets, frames)

            if writer is not None:
                annotated = frame.copy()
                for track, det in pairs:
                    x, y, w, h = det.box
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 255), 2)
                    cv2.putText(
                        annotated,
                        f"#{track.track_id}",
                        (x, max(y - 8, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 255),
                        2,
                        cv2.LINE_AA,
                    )
                writer.write(annotated)

            frames += 1
            if frames % 500 == 0:
                print(f"  {frames}/{total} frames…", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    all_tracks = list(tracker.finish())
    people = [t for t in all_tracks if t.frames_seen >= args.min_frames]
    report = build_report(people, all_tracks, frames, fps)

    elapsed = max(time.time() - started, 1e-6)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print()
        print(report.summary())
        print(f"\nProcessed at {frames / elapsed:.1f} fps")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    """Count people crossing a line — the footfall measure.

    Robust where unique-person counting is not: a person fragmented into
    several tracks still crosses the line once, so the count degrades
    gracefully instead of inflating.
    """
    cap = cv2.VideoCapture(args.path)
    if not cap.isOpened():
        print(f"Cannot open video: {args.path}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    diagonal = (width**2 + height**2) ** 0.5

    try:
        line = CountingLine.parse(args.line, width, height, positive_is=args.positive_is)
    except ValueError as exc:
        print(exc)
        return 1

    detector = PersonDetector()
    tracker = PersonTracker(frame_diagonal=diagonal)
    counter = CrossingCounter(line=line)
    motion = CameraMotionEstimator(enabled=not args.no_motion_compensation)

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )

    print(f"Video   {args.path}  {width}x{height} @ {fps:.0f}fps")
    print(f"Line    {line.start} to {line.end}   (positive side = '{line.positive_is}')")
    print(f"Motion  compensation {'off' if args.no_motion_compensation else 'on'}\n")

    frames = 0
    motion_total = 0.0
    started = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            dets = detector.detect(frame, confidence_threshold=args.confidence)

            transform = motion.update(frame, dets)
            motion_total += transform.magnitude
            tracker.compensate(transform)

            pairs = tracker.update(dets, frames)

            for track, det in pairs:
                counter.update(track.track_id, det.centroid, frames)

            if writer is not None:
                annotated = draw(frame, [d for _, d in pairs])
                p1 = (int(line.start[0]), int(line.start[1]))
                p2 = (int(line.end[0]), int(line.end[1]))
                cv2.line(annotated, p1, p2, (0, 0, 255), 2)
                cv2.putText(
                    annotated,
                    f"in {counter.total_in}  out {counter.total_out}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)

            frames += 1
            if frames % 500 == 0:
                print(f"  {frames}/{total} frames…", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed = max(time.time() - started, 1e-6)

    if args.json:
        payload = counter.to_dict()
        payload["frames"] = frames
        payload["fps"] = fps
        print(json.dumps(payload, indent=2))
    else:
        print()
        print(counter.summary(fps=fps))
        print(f"\nMean camera motion  {motion_total / max(frames, 1):.2f} px/frame")
        print(f"Processed           {frames / elapsed:.1f} fps")
    if args.output:
        print(f"Wrote {args.output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="human-detection", description=__doc__)
    parser.add_argument("-c", "--confidence", type=float, default=0.5)
    parser.add_argument("--all-classes", action="store_true", help="report all 80 COCO classes")
    parser.add_argument("--raw", action="store_true", help="skip duplicate suppression")
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", help="detect in a still image")
    p_img.add_argument("path")
    p_img.add_argument("-o", "--output")
    p_img.set_defaults(func=cmd_image)

    p_vid = sub.add_parser("video", help="detect across a video file")
    p_vid.add_argument("path")
    p_vid.add_argument("-o", "--output")
    p_vid.add_argument("--show", action="store_true", help="display while processing")
    p_vid.set_defaults(func=cmd_video)

    p_cam = sub.add_parser("webcam", help="live detection from a camera")
    p_cam.add_argument("-d", "--device", type=int, default=0)
    p_cam.set_defaults(func=cmd_webcam)

    p_an = sub.add_parser("analyse", help="track people and report audience metrics")
    p_an.add_argument("path")
    p_an.add_argument("-o", "--output", help="write a video annotated with track ids")
    p_an.add_argument("--json", action="store_true")
    p_an.add_argument(
        "--min-frames",
        type=int,
        default=None,
        help="frames a track must survive to count as a person",
    )
    p_an.set_defaults(func=cmd_analyse)

    p_count = sub.add_parser("count", help="count people crossing a line (footfall)")
    p_count.add_argument("path", help="any video file")
    p_count.add_argument(
        "-l",
        "--line",
        default="horizontal",
        help="'horizontal', 'vertical:0.6', or 'x1,y1,x2,y2' (default: horizontal mid-frame)",
    )
    p_count.add_argument(
        "--positive-is",
        choices=["in", "out"],
        default="in",
        help="label for crossings toward the positive side of the line",
    )
    p_count.add_argument("-o", "--output", help="write an annotated video")
    p_count.add_argument("--json", action="store_true")
    p_count.add_argument(
        "--no-motion-compensation",
        action="store_true",
        help="disable camera-motion compensation (useful for A/B comparison)",
    )
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args()
    if getattr(args, "min_frames", None) is None and args.command == "analyse":
        from humandetect.config import TRACK_MIN_FRAMES

        args.min_frames = TRACK_MIN_FRAMES
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

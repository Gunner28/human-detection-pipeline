"""Run the pipeline over MOT17 and score it with TrackEval.

Two halves, deliberately separate: `run_sequence` produces result files
and touches no metrics; `evaluate` reads result files and runs no
detector. Keeping them apart means a scoring change never silently
requires a four-minute re-detection, and a detector change is always
scored by identical code.

The metrics are not reimplemented. TrackEval is the official
implementation behind the MOTChallenge leaderboard; its preprocessing
step — dropping detections that matched distractors, static persons and
other ignore classes before counting false positives — is where hand-made
evaluators usually diverge without anyone noticing.

    ./scripts/setup_trackeval.sh    once, clones and patches TrackEval

Detections here are *private* (this project's own detector), not the
public det/det.txt that ships with MOT17. That is a separate MOTChallenge
leaderboard, so published numbers are only comparable within the same mode.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from . import mot17
from .config import CONFIDENCE_THRESHOLD, PROJECT_ROOT, TRACK_MIN_FRAMES
from .motion import CameraMotionEstimator
from .tracking import PersonTracker

TRACKEVAL_PATH = PROJECT_ROOT / "external" / "TrackEval"
RESULTS_ROOT = PROJECT_ROOT / "outputs" / "mot17"
EVAL_ROOT = PROJECT_ROOT / "outputs" / "mot17_eval"

# Reported for every run. HOTA is the headline; DetA and AssA are its two
# factors, and reporting them separately is what lets an improvement be
# attributed to detection or to association rather than guessed at.
HEADLINE_FIELDS = (
    ("HOTA", "HOTA"),
    ("HOTA", "DetA"),
    ("HOTA", "AssA"),
    ("Identity", "IDF1"),
    ("CLEAR", "MOTA"),
    ("CLEAR", "MOTP"),
)
COUNT_FIELDS = (
    ("CLEAR", "CLR_TP"),
    ("CLEAR", "CLR_FN"),
    ("CLEAR", "CLR_FP"),
    ("CLEAR", "IDSW"),
    ("CLEAR", "Frag"),
)


@dataclass
class BenchmarkConfig:
    """Everything that changes a run's numbers, in one object.

    Anything not named here is a repo default, so two runs that differ
    only in this object differ only in what it says.
    """

    backend: str = "yolo"
    imgsz: int = 1280
    confidence: float = CONFIDENCE_THRESHOLD
    min_frames: int = TRACK_MIN_FRAMES
    compensate: bool = False
    cap_imgsz_to_native: bool = True
    # None uses the configured default. Above 1.0 disables the containment
    # half of duplicate suppression, since containment can never exceed 1.
    containment: float | None = None

    def effective_imgsz(self, info: mot17.SeqInfo) -> int:
        """Inference size for this sequence, never above its native size.

        Upscaling past native resolution invents no detail but does invent
        false positives: MOT17-05 (640x480) at imgsz=1280 scored 1,546
        false positives against 673 at native size, costing 22.6 points of
        MOTA. YOLO wants a multiple of 32.
        """
        if self.backend == "ssd" or not self.cap_imgsz_to_native:
            return self.imgsz
        native = max(info.width, info.height)
        capped = min(self.imgsz, native)
        return max(32, int(math.ceil(capped / 32) * 32))

    def describe(self) -> str:
        size = "320 (fixed)" if self.backend == "ssd" else f"{self.imgsz} max"
        return (
            f"backend={self.backend} imgsz={size} confidence={self.confidence} "
            f"min_frames={self.min_frames} compensate={self.compensate}"
        )


@dataclass
class RunStats:
    """What one sequence run produced. Every field is a count, not a score."""

    sequence: str
    frames: int
    imgsz: int
    detections: int
    rows_written: int
    tracks_total: int
    tracks_kept: int
    seconds: float

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


def _build_detector(config: BenchmarkConfig, imgsz: int):
    """Return a detect(frame) -> list[Detection] closure."""
    if config.backend == "ssd":
        from .detector import PersonDetector

        detector = PersonDetector()
        return lambda frame: detector.detect(
            frame,
            confidence_threshold=config.confidence,
            containment_threshold=config.containment,
        )

    from .backends import load_backend

    backend = load_backend(config.backend, imgsz=imgsz)
    return lambda frame: backend.detect(
        frame,
        confidence_threshold=config.confidence,
        containment_threshold=config.containment,
    )


def run_sequence(
    seq_dir: Path,
    out_path: Path,
    config: BenchmarkConfig | None = None,
    *,
    progress: bool = False,
) -> RunStats:
    """Detect and track one MOT17 sequence, writing MOTChallenge output."""
    config = config or BenchmarkConfig()
    info = mot17.read_seqinfo(seq_dir)
    frames = mot17.frame_paths(seq_dir, info)
    imgsz = config.effective_imgsz(info)

    detect = _build_detector(config, imgsz)
    tracker = PersonTracker(frame_diagonal=math.hypot(info.width, info.height))
    motion = CameraMotionEstimator() if config.compensate else None

    rows: list[tuple[int, int, tuple[int, int, int, int], float]] = []
    detection_count = 0
    previous_detections: list = []
    started = time.perf_counter()

    for frame_index, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise mot17.MOT17Error(f"{seq_dir.name}: cannot read {frame_path.name}")

        if motion is not None:
            # The previous frame's boxes mask feature points that sit on
            # people, so the estimate follows the camera, not the crowd.
            tracker.compensate(motion.update(frame, previous_detections))

        detections = detect(frame)
        previous_detections = detections
        detection_count += len(detections)

        for track, detection in tracker.update(detections, frame_index):
            rows.append(
                (frame_index, track.track_id, detection.box, detection.confidence)
            )

        if progress and frame_index % 100 == 0:
            print(f"    {seq_dir.name} {frame_index}/{info.length}", flush=True)

    elapsed = time.perf_counter() - started
    all_tracks = tracker.finish()

    # Short tracks are detector flicker rather than people. The pipeline
    # already applies this filter to its own counts, so applying it here
    # keeps the benchmark measuring what the product actually reports.
    kept_ids = {t.track_id for t in all_tracks if t.frames_seen >= config.min_frames}
    kept_rows = [r for r in rows if r[1] in kept_ids]
    written = mot17.write_results(kept_rows, out_path, seq_length=info.length)

    return RunStats(
        sequence=seq_dir.name,
        frames=info.length,
        imgsz=imgsz,
        detections=detection_count,
        rows_written=written,
        tracks_total=len(all_tracks),
        tracks_kept=len(kept_ids),
        seconds=elapsed,
    )


def run(
    scenes: list[str],
    tracker_name: str,
    config: BenchmarkConfig | None = None,
    *,
    verbose: bool = True,
) -> list[RunStats]:
    """Run several sequences into outputs/mot17/<tracker_name>/data/."""
    config = config or BenchmarkConfig()
    out_dir = RESULTS_ROOT / tracker_name / "data"
    results = []
    if verbose:
        print(config.describe() + "\n")
    for scene in scenes:
        seq_dir = mot17.resolve(scene)
        if not seq_dir.is_dir():
            raise mot17.MOT17Error(f"{seq_dir} not found")
        stats = run_sequence(seq_dir, out_dir / f"{seq_dir.name}.txt", config)
        results.append(stats)
        if verbose:
            print(
                f"  {stats.sequence:<18} imgsz {stats.imgsz:>4}  "
                f"{stats.frames:>4} frames  {stats.detections:>6} detections  "
                f"{stats.tracks_kept:>3}/{stats.tracks_total:<3} tracks  "
                f"{stats.fps:>5.1f} fps"
            )
    return results


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def _import_trackeval():
    if not TRACKEVAL_PATH.is_dir():
        raise RuntimeError(
            f"TrackEval not found at {TRACKEVAL_PATH}. "
            f"Run ./scripts/setup_trackeval.sh"
        )
    if str(TRACKEVAL_PATH) not in sys.path:
        sys.path.insert(0, str(TRACKEVAL_PATH))
    import trackeval  # noqa: PLC0415

    return trackeval


def field_value(field_ref: tuple[str, str], scores: dict) -> float:
    """HOTA fields are arrays over localisation thresholds; others scalars."""
    family, name = field_ref
    value = scores[family][name]
    return float(value.mean()) if hasattr(value, "mean") else float(value)


@dataclass
class EvalResult:
    """Scores for one tracker run, per sequence and combined."""

    tracker: str
    sequences: list[str]
    per_sequence: dict[str, dict] = field(default_factory=dict)
    combined: dict = field(default_factory=dict)

    def score(self, field_ref: tuple[str, str], sequence: str | None = None) -> float:
        scores = self.combined if sequence is None else self.per_sequence[sequence]
        return field_value(field_ref, scores)

    def headline(self, sequence: str | None = None) -> dict[str, float]:
        """HOTA, DetA, AssA, IDF1, MOTA, MOTP as percentages."""
        return {
            name: self.score(ref, sequence) * 100
            for ref in HEADLINE_FIELDS
            for name in (ref[1],)
        }

    def counts(self, sequence: str | None = None) -> dict[str, int]:
        """TP, FN, FP, ID switches and fragmentations as raw counts."""
        return {
            name: int(self.score(ref, sequence))
            for ref in COUNT_FIELDS
            for name in (ref[1],)
        }

    def table(self) -> str:
        lines = [
            f"{'sequence':<20}" + "".join(f"{n:>8}" for _f, n in HEADLINE_FIELDS)
        ]
        for name in self.sequences:
            row = "".join(
                f"{self.score(f, name) * 100:>8.2f}" for f in HEADLINE_FIELDS
            )
            lines.append(f"{name:<20}{row}")
        row = "".join(f"{self.score(f) * 100:>8.2f}" for f in HEADLINE_FIELDS)
        lines.append(f"{'COMBINED':<20}{row}")
        return "\n".join(lines)

    def summary(self) -> str:
        tp = self.score(("CLEAR", "CLR_TP"))
        fn = self.score(("CLEAR", "CLR_FN"))
        fp = self.score(("CLEAR", "CLR_FP"))
        gt = tp + fn
        return (
            f"Recall {100 * tp / gt:.1f}% ({tp:.0f} of {gt:.0f} ground-truth boxes), "
            f"precision {100 * tp / (tp + fp):.1f}% ({fp:.0f} false positives)."
        )


def evaluate(
    tracker_name: str,
    sequences: list[str] | None = None,
    *,
    quiet: bool = True,
) -> EvalResult:
    """Score a run in outputs/mot17/<tracker_name>/ against MOT17 train gt."""
    trackeval = _import_trackeval()

    data_dir = RESULTS_ROOT / tracker_name / "data"
    if not data_dir.is_dir():
        raise RuntimeError(f"No results at {data_dir} — run the benchmark first")

    if sequences:
        names = [mot17.resolve(s).name for s in sequences]
    else:
        names = sorted(p.stem for p in data_dir.glob("*.txt"))
    if not names:
        raise RuntimeError(f"No result files in {data_dir}")

    for name in names:
        if not (data_dir / f"{name}.txt").is_file():
            raise RuntimeError(f"Missing result file {data_dir / name}.txt")
        if not (mot17.MOT_ROOT / "train" / name / "gt" / "gt.txt").is_file():
            raise RuntimeError(f"Missing ground truth for {name}")

    seq_info = {
        name: mot17.read_seqinfo(mot17.MOT_ROOT / "train" / name).length
        for name in names
    }

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_CONFIG": False,
        "PRINT_RESULTS": False,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
        "OUTPUT_FOLDER": str(EVAL_ROOT),
    })

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(mot17.MOT_ROOT / "train"),
        "TRACKERS_FOLDER": str(RESULTS_ROOT),
        "TRACKERS_TO_EVAL": [tracker_name],
        "BENCHMARK": "MOT17",
        "SPLIT_TO_EVAL": "train",
        "SKIP_SPLIT_FOL": True,  # gt and results are pointed at directly
        "SEQ_INFO": seq_info,    # so no seqmap file is needed
        "DO_PREPROC": True,      # ignore-region handling — do not turn off
        "PRINT_CONFIG": False,
    })

    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
        "PRINT_CONFIG": False,
    }
    metrics = [
        trackeval.metrics.HOTA(metrics_config),
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]

    if quiet:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            output, _ = trackeval.Evaluator(eval_config).evaluate(
                [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics
            )
    else:
        output, _ = trackeval.Evaluator(eval_config).evaluate(
            [trackeval.datasets.MotChallenge2DBox(dataset_config)], metrics
        )

    per_tracker = output["MotChallenge2DBox"][tracker_name]
    return EvalResult(
        tracker=tracker_name,
        sequences=names,
        per_sequence={n: per_tracker[n]["pedestrian"] for n in names},
        combined=per_tracker["COMBINED_SEQ"]["pedestrian"],
    )

"""MOT17 benchmark data: reading it, checking it, writing results for it.

Why this module exists
----------------------
Before MOT17, this project measured itself against frames labelled by
hand. That put a ceiling on the work: a few hundred labelled frames, one
annotator, and no way to compare the result with anyone else's. MOT17
ships 5,316 frames with 112,297 ground-truth pedestrian boxes and a track
id on every one of them, which is both larger than any hand-labelled set
and directly comparable with published work.

The dataset is 5.6 GB and is not in this repository. Fetch it from
motchallenge.net and unzip it to ./MOT17, then run `verify()` — nothing
else here trusts the copy on disk until that passes.

Layout
------
    MOT17/train/MOT17-02-FRCNN/
        img1/000001.jpg ...   frames, 1-indexed, six digits
        det/det.txt           public detections (unused: we detect ourselves)
        gt/gt.txt             ground truth (train split only)
        seqinfo.ini           fps, resolution, frame count

The 21 train folders are 7 scenes duplicated once per public detector
(DPM, FRCNN, SDP). The frames and the ground truth are identical across
the three copies; only det.txt differs. Averaging a metric over all 21
therefore triple-counts 7 videos, which is the most common way to produce
a MOT17 number that does not match anyone else's.

Reference: Milan et al., "MOT16: A Benchmark for Multi-Object Tracking"
(2016); Dendorfer et al., MOTChallenge.
"""
from __future__ import annotations

import configparser
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import PROJECT_ROOT

MOT_ROOT = PROJECT_ROOT / "MOT17"

DETECTORS = ("DPM", "FRCNN", "SDP")
TRAIN_SCENES = ("02", "04", "05", "09", "10", "11", "13")
TEST_SCENES = ("01", "03", "06", "07", "08", "12", "14")

# Frames are identical across the three detector copies, so for private
# detections the choice is arbitrary. FRCNN is the canonical one here.
CANONICAL_VARIANT = "FRCNN"

# gt.txt column 8. Class 1 is the only class scored. The others are either
# ignore regions or non-target objects: a detection landing on one counts
# as neither a hit nor a false positive. Dropping this distinction is the
# usual reason a home-made evaluator disagrees with the official numbers.
CLASS_NAMES = {
    1: "pedestrian",
    2: "person on vehicle",
    3: "car",
    4: "bicycle",
    5: "motorbike",
    6: "non-motorised vehicle",
    7: "static person",
    8: "distractor",
    9: "occluder",
    10: "occluder on ground",
    11: "occluder full",
    12: "reflection",
}
EVALUATED_CLASS = 1


class MOT17Error(RuntimeError):
    """The dataset on disk is missing, incomplete, or internally inconsistent."""


@dataclass(frozen=True)
class SeqInfo:
    """What seqinfo.ini declares about a sequence."""

    name: str
    length: int
    width: int
    height: int
    frame_rate: int
    ext: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class GtBox:
    """One ground-truth box. Fields are gt.txt's columns, in order."""

    frame: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    conf: int          # 0 = ignore this row entirely, 1 = use it
    class_id: int
    visibility: float  # 0..1, fraction of the person not occluded

    @property
    def is_evaluated(self) -> bool:
        return self.conf == 1 and self.class_id == EVALUATED_CLASS


def resolve(scene: str, split: str = "train", root: Path | None = None) -> Path:
    """Path for a scene given as '02', 'MOT17-02', or 'MOT17-02-SDP'."""
    root = root or MOT_ROOT
    name = scene if scene.startswith("MOT17-") else f"MOT17-{scene}"
    if not name.endswith(tuple(f"-{d}" for d in DETECTORS)):
        name = f"{name}-{CANONICAL_VARIANT}"
    return root / split / name


def train_sequences(root: Path | None = None) -> list[Path]:
    """The 7 canonical train scenes, one folder each."""
    return [resolve(s, "train", root) for s in TRAIN_SCENES]


def read_seqinfo(seq_dir: Path) -> SeqInfo:
    ini = seq_dir / "seqinfo.ini"
    if not ini.is_file():
        raise MOT17Error(f"{seq_dir.name}: seqinfo.ini missing")
    parser = configparser.ConfigParser()
    parser.read(ini)
    section = parser["Sequence"]
    return SeqInfo(
        name=section["name"],
        length=int(section["seqLength"]),
        width=int(section["imWidth"]),
        height=int(section["imHeight"]),
        frame_rate=int(section["frameRate"]),
        ext=section["imExt"],
    )


def frame_paths(seq_dir: Path, info: SeqInfo | None = None) -> list[Path]:
    """Frame files in order, checked against the declared length."""
    info = info or read_seqinfo(seq_dir)
    frames = sorted((seq_dir / "img1").glob(f"*{info.ext}"))
    if len(frames) != info.length:
        raise MOT17Error(
            f"{seq_dir.name}: {len(frames)} frames on disk, "
            f"seqinfo declares {info.length}"
        )
    return frames


def read_gt(seq_dir: Path) -> list[GtBox]:
    """Parse gt/gt.txt. Returns every row, evaluated or not."""
    path = seq_dir / "gt" / "gt.txt"
    if not path.is_file():
        raise MOT17Error(f"{seq_dir.name}: gt/gt.txt missing (test split has none)")

    boxes: list[GtBox] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            raise MOT17Error(
                f"{seq_dir.name} gt.txt line {lineno}: "
                f"expected 9 columns, got {len(parts)}"
            )
        boxes.append(
            GtBox(
                frame=int(parts[0]),
                track_id=int(parts[1]),
                left=float(parts[2]),
                top=float(parts[3]),
                width=float(parts[4]),
                height=float(parts[5]),
                conf=int(float(parts[6])),
                class_id=int(float(parts[7])),
                visibility=float(parts[8]),
            )
        )
    return boxes


def write_results(
    rows: list[tuple[int, int, tuple[int, int, int, int], float]],
    out_path: Path,
    *,
    seq_length: int,
) -> int:
    """Write tracker output in MOTChallenge format and assert it is sane.

        frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1

    The trailing -1s are the 3D fields, unused in 2D benchmarks. Returns
    the number of rows written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for frame, track_id, (x, y, w, h), conf in sorted(rows):
            handle.write(f"{frame},{track_id},{x},{y},{w},{h},{conf:.4f},-1,-1,-1\n")

    written = [
        line for line in out_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(written) != len(rows):
        raise MOT17Error(
            f"{out_path.name}: wrote {len(written)} lines for {len(rows)} rows"
        )

    seen: dict[tuple[int, int], int] = defaultdict(int)
    for line in written:
        parts = line.split(",")
        frame, track_id = int(parts[0]), int(parts[1])
        if not 1 <= frame <= seq_length:
            raise MOT17Error(
                f"{out_path.name}: frame {frame} outside 1..{seq_length}"
            )
        seen[(frame, track_id)] += 1
    duplicated = [key for key, count in seen.items() if count > 1]
    if duplicated:
        raise MOT17Error(
            f"{out_path.name}: track id repeated within a frame, "
            f"first at frame {duplicated[0][0]} id {duplicated[0][1]}"
        )
    return len(written)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------
@dataclass
class SequenceStats:
    """What verify() found in one sequence."""

    name: str
    length: int
    resolution: str
    frame_rate: int
    gt_rows: int = 0
    gt_evaluated: int = 0
    gt_ids: int = 0
    class_counts: Counter = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()


def verify_sequence(seq_dir: Path, *, expect_gt: bool) -> SequenceStats:
    """Check one sequence folder end to end. Raises MOT17Error on any fault."""
    info = read_seqinfo(seq_dir)
    if info.name != seq_dir.name:
        raise MOT17Error(f"{seq_dir.name}: seqinfo names itself {info.name!r}")

    frames = frame_paths(seq_dir, info)
    numbers = sorted(int(f.stem) for f in frames)
    if numbers != list(range(1, info.length + 1)):
        raise MOT17Error(
            f"{seq_dir.name}: frame numbering is not 1..{info.length} contiguous"
        )
    if any(f.stat().st_size == 0 for f in frames):
        raise MOT17Error(f"{seq_dir.name}: at least one zero-byte frame")

    det = seq_dir / "det" / "det.txt"
    if not det.is_file() or det.stat().st_size == 0:
        raise MOT17Error(f"{seq_dir.name}: det/det.txt missing or empty")

    stats = SequenceStats(
        name=seq_dir.name,
        length=info.length,
        resolution=info.resolution,
        frame_rate=info.frame_rate,
    )

    gt_path = seq_dir / "gt" / "gt.txt"
    if not expect_gt:
        if gt_path.exists():
            raise MOT17Error(f"{seq_dir.name}: test sequence unexpectedly has gt")
        return stats

    boxes = read_gt(seq_dir)
    if not boxes:
        raise MOT17Error(f"{seq_dir.name}: gt.txt has no rows")

    for box in boxes:
        if not 1 <= box.frame <= info.length:
            raise MOT17Error(
                f"{seq_dir.name}: gt frame {box.frame} outside 1..{info.length}"
            )
        if box.track_id < 1:
            raise MOT17Error(f"{seq_dir.name}: gt track id {box.track_id} below 1")
        if box.width <= 0 or box.height <= 0:
            raise MOT17Error(
                f"{seq_dir.name}: non-positive box size on frame {box.frame}"
            )
        if box.conf not in (0, 1):
            raise MOT17Error(
                f"{seq_dir.name}: gt conf flag {box.conf} on frame {box.frame}"
            )
        if box.class_id not in CLASS_NAMES:
            raise MOT17Error(
                f"{seq_dir.name}: unknown gt class {box.class_id} "
                f"on frame {box.frame}"
            )
        if not 0.0 <= box.visibility <= 1.0:
            raise MOT17Error(
                f"{seq_dir.name}: visibility {box.visibility} outside 0..1"
            )
        stats.class_counts[box.class_id] += 1

    evaluated = [b for b in boxes if b.is_evaluated]
    stats.gt_rows = len(boxes)
    stats.gt_evaluated = len(evaluated)
    stats.gt_ids = len({b.track_id for b in evaluated})

    # A track id twice in one frame would make the ground-truth partition
    # ill-defined, and the identity matching behind IDF1 ambiguous.
    per_frame: dict[int, set[int]] = defaultdict(set)
    for box in evaluated:
        if box.track_id in per_frame[box.frame]:
            raise MOT17Error(
                f"{seq_dir.name}: track {box.track_id} twice in frame {box.frame}"
            )
        per_frame[box.frame].add(box.track_id)

    return stats


def verify(root: Path | None = None) -> dict[str, list[SequenceStats]]:
    """Check the whole dataset. Raises MOT17Error on the first fault found."""
    root = root or MOT_ROOT
    if not root.is_dir():
        raise MOT17Error(
            f"MOT17 not found at {root}. Download MOT17.zip from "
            f"motchallenge.net and unzip it there."
        )

    expected = {
        "train": [f"MOT17-{s}-{d}" for s in TRAIN_SCENES for d in DETECTORS],
        "test": [f"MOT17-{s}-{d}" for s in TEST_SCENES for d in DETECTORS],
    }

    found_stats: dict[str, list[SequenceStats]] = {}
    for split, names in expected.items():
        split_dir = root / split
        if not split_dir.is_dir():
            raise MOT17Error(f"{split}/ missing under {root}")
        on_disk = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        if on_disk != sorted(names):
            raise MOT17Error(
                f"{split}/: expected {len(names)} sequences, found {len(on_disk)}. "
                f"Missing: {sorted(set(names) - set(on_disk))}"
            )
        found_stats[split] = [
            verify_sequence(split_dir / name, expect_gt=(split == "train"))
            for name in sorted(names)
        ]

    # The three detector copies of a scene must agree, since they are the
    # same video. This is the fact that makes averaging over 21 a triple-count.
    for scene in TRAIN_SCENES:
        variants = [
            next(s for s in found_stats["train"] if s.name == f"MOT17-{scene}-{d}")
            for d in DETECTORS
        ]
        if len({v.length for v in variants}) != 1 or len({v.gt_rows for v in variants}) != 1:
            raise MOT17Error(
                f"MOT17-{scene}: detector variants disagree on length or gt rows"
            )

    return found_stats

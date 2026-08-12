"""Detection evaluation against labelled frames.

Two levels of evaluation, because two levels of ground truth are
practical to produce:

**Box-level** — precision, recall, F1 with greedy IoU matching. The proper
measure, but it needs boxes drawn by hand for every person.

**Count-level** — how many people the detector reports versus how many are
actually there. Weaker, but it is what a single annotator can produce
reliably, and it directly answers the question this project cares about:
is the headcount too high or too low, and by how much?

Both apply a **minimum height filter**. Distant pedestrians in 854x480
footage are a handful of pixels tall and cannot be counted reliably by eye,
so they are excluded from ground truth *and* predictions rather than
silently counted as detector errors. COCO does the same thing with its
size categories; crowd datasets use ignore regions for the same reason.
Every metric therefore reads "for people at least N pixels tall", and that
qualifier is not optional when quoting the result.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .detector import Detection, _iou_and_containment

DEFAULT_MIN_HEIGHT = 60  # pixels, in the source resolution


# --------------------------------------------------------------------------
# Box-level
# --------------------------------------------------------------------------
@dataclass
class FrameResult:
    """Matching outcome for one frame."""

    frame_index: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0


@dataclass
class BoxReport:
    """Aggregate box-level metrics."""

    iou_threshold: float
    min_height: int
    frames: list[FrameResult] = field(default_factory=list)

    @property
    def true_positives(self) -> int:
        return sum(f.true_positives for f in self.frames)

    @property
    def false_positives(self) -> int:
        return sum(f.false_positives for f in self.frames)

    @property
    def false_negatives(self) -> int:
        return sum(f.false_negatives for f in self.frames)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def summary(self) -> str:
        return "\n".join(
            [
                f"Box-level, IoU>={self.iou_threshold}, people >= {self.min_height}px tall",
                f"  frames        {len(self.frames)}",
                f"  true pos      {self.true_positives}",
                f"  false pos     {self.false_positives}",
                f"  false neg     {self.false_negatives}",
                f"  precision     {self.precision:.1%}",
                f"  recall        {self.recall:.1%}",
                f"  F1            {self.f1:.3f}",
            ]
        )


def match_frame(
    predictions: list[Detection],
    ground_truth: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
    min_height: int = DEFAULT_MIN_HEIGHT,
    frame_index: int = 0,
) -> FrameResult:
    """Greedy IoU matching of predictions to ground-truth boxes."""
    preds = sorted(
        (p for p in predictions if p.box[3] >= min_height),
        key=lambda p: p.confidence,
        reverse=True,
    )
    truths = [g for g in ground_truth if g[3] >= min_height]

    matched: set[int] = set()
    true_positives = 0

    for pred in preds:
        best_iou, best_idx = 0.0, -1
        for idx, truth in enumerate(truths):
            if idx in matched:
                continue
            iou, _ = _iou_and_containment(pred.box, truth)
            if iou > best_iou:
                best_iou, best_idx = iou, idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched.add(best_idx)
            true_positives += 1

    return FrameResult(
        frame_index=frame_index,
        true_positives=true_positives,
        false_positives=len(preds) - true_positives,
        false_negatives=len(truths) - true_positives,
    )


# --------------------------------------------------------------------------
# Count-level
# --------------------------------------------------------------------------
@dataclass
class CountReport:
    """How detector headcounts compare with labelled counts."""

    min_height: int
    confidence: float
    frame_indices: list[int] = field(default_factory=list)
    predicted: list[int] = field(default_factory=list)
    actual: list[int] = field(default_factory=list)

    @property
    def errors(self) -> list[int]:
        return [p - a for p, a in zip(self.predicted, self.actual)]

    @property
    def mean_absolute_error(self) -> float:
        errs = self.errors
        return sum(abs(e) for e in errs) / len(errs) if errs else 0.0

    @property
    def bias(self) -> float:
        """Positive means the detector over-counts on average."""
        errs = self.errors
        return sum(errs) / len(errs) if errs else 0.0

    @property
    def total_predicted(self) -> int:
        return sum(self.predicted)

    @property
    def total_actual(self) -> int:
        return sum(self.actual)

    @property
    def ratio(self) -> float:
        """Detected people as a fraction of real people. 1.0 is perfect."""
        return self.total_predicted / self.total_actual if self.total_actual else 0.0

    def summary(self) -> str:
        lines = [
            f"Count-level, confidence>={self.confidence}, people >= {self.min_height}px tall",
            f"  frames            {len(self.actual)}",
            f"  labelled people   {self.total_actual}",
            f"  detected people   {self.total_predicted}",
            f"  detected / actual {self.ratio:.2f}x",
            f"  mean abs error    {self.mean_absolute_error:.2f} people/frame",
            f"  bias              {self.bias:+.2f} people/frame",
            "",
            f"  {'frame':>7} {'actual':>7} {'detected':>9} {'error':>6}",
        ]
        for idx, actual, pred in zip(self.frame_indices, self.actual, self.predicted):
            lines.append(f"  {idx:>7} {actual:>7} {pred:>9} {pred - actual:>+6}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "min_height": self.min_height,
            "confidence": self.confidence,
            "frames": len(self.actual),
            "labelled_people": self.total_actual,
            "detected_people": self.total_predicted,
            "ratio": round(self.ratio, 3),
            "mean_absolute_error": round(self.mean_absolute_error, 3),
            "bias": round(self.bias, 3),
            "per_frame": [
                {"frame": i, "actual": a, "detected": p}
                for i, a, p in zip(self.frame_indices, self.actual, self.predicted)
            ],
        }


def load_ground_truth(path: str | Path) -> dict:
    """Load a hand-labelled set: outputs/ground_truth.json.

    Kept for the count-level question MOT17 cannot answer — whether the
    headcount on this project's own footage is right. See docs/FINDINGS.md
    #5 for why that particular set is stamped unreliable, and #10 onward
    for the benchmark that replaced it for measuring detection quality.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))

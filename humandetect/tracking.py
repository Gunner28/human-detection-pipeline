"""Multi-object tracking: turning per-frame boxes into persistent people.

Why this exists
---------------
Detection alone cannot answer the question the project was set up to
answer. Running the detector over a 2.3-minute clip produced 20,765 person
boxes across 4,094 frames — but that is not 20,765 people. It is a much
smaller number of people, each detected once per frame they appear in.
Someone who walks across the shot for four seconds contributes ~120 boxes.

"5.07 people per frame" is a debug print. "47 unique people passed
through, average dwell 3.2 seconds" is an audience metric. Tracking is
what converts the first into the second.

How it works
------------
Greedy IoU matching with a centroid-distance fallback, in the spirit of
SORT but without its Kalman filter or its scipy dependency:

1. Each frame, compute IoU between every open track and every detection.
2. Assign the highest-scoring pairs first, greedily.
3. Detections that match nothing start a new track.
4. Tracks that match nothing age; after `TRACK_MAX_MISSING` frames they
   close. That tolerance is what lets a person survive walking behind a
   sign without being counted twice.

The trade-off is deliberate: no motion model, so two people crossing paths
can swap identities. A Kalman filter would help, and is the obvious next
upgrade — but this version is explainable end to end, which matters more
while the numbers are still being validated.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import (
    TRACK_DISTANCE_FRACTION,
    TRACK_IOU_THRESHOLD,
    TRACK_MAX_MISSING,
    TRACK_MIN_FRAMES,
)
from .detector import Detection, _iou_and_containment


@dataclass
class Track:
    """One person followed across frames."""

    track_id: int
    box: tuple[int, int, int, int]
    first_frame: int
    last_frame: int
    frames_seen: int = 1
    missing: int = 0
    confidence_sum: float = 0.0
    centroids: list[tuple[float, float]] = field(default_factory=list)

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.frames_seen if self.frames_seen else 0.0

    @property
    def span_frames(self) -> int:
        """Frames between first and last sighting, including gaps."""
        return self.last_frame - self.first_frame + 1

    def distance_travelled(self) -> float:
        """Total path length in pixels — separates walkers from loiterers."""
        total = 0.0
        for (x1, y1), (x2, y2) in zip(self.centroids, self.centroids[1:]):
            total += math.hypot(x2 - x1, y2 - y1)
        return total

    def update(self, detection: Detection, frame_index: int) -> None:
        self.box = detection.box
        self.last_frame = frame_index
        self.frames_seen += 1
        self.missing = 0
        self.confidence_sum += detection.confidence
        self.centroids.append(detection.centroid)


class PersonTracker:
    """Assigns and maintains stable ids for detected people."""

    def __init__(
        self,
        iou_threshold: float = TRACK_IOU_THRESHOLD,
        max_missing: int = TRACK_MAX_MISSING,
        distance_fraction: float = TRACK_DISTANCE_FRACTION,
        frame_diagonal: float | None = None,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.distance_fraction = distance_fraction
        self.frame_diagonal = frame_diagonal

        self.active: list[Track] = []
        self.closed: list[Track] = []
        self._next_id = 1

    # --- internals -------------------------------------------------------
    def _max_distance(self) -> float:
        if self.frame_diagonal is None:
            return float("inf")
        return self.frame_diagonal * self.distance_fraction

    def _score(self, track: Track, detection: Detection) -> float:
        """Match score in [0, 1]; 0 means "not a candidate".

        IoU is preferred. When boxes do not overlap at all, fall back to
        centroid proximity so a fast walker is not handed a new identity
        every few frames.
        """
        iou, _ = _iou_and_containment(track.box, detection.box)
        if iou >= self.iou_threshold:
            return iou

        limit = self._max_distance()
        if limit == float("inf"):
            return 0.0

        tx, ty = track.centroid
        dx, dy = detection.centroid
        distance = math.hypot(dx - tx, dy - ty)
        if distance > limit:
            return 0.0
        # Scale below the IoU band so genuine overlaps always win.
        return (1.0 - distance / limit) * self.iou_threshold * 0.9

    # --- public ----------------------------------------------------------
    def compensate(self, transform) -> None:
        """Move open tracks into the current frame's coordinates.

        Called before `update()` when camera-motion compensation is on, so
        matching compares like with like. Without it, a stationary person
        filmed by a moving camera looks like a new person every few frames
        — the dominant cause of inflated counts on handheld footage.
        """
        if transform is None or transform.is_identity:
            return
        for track in self.active:
            track.box = transform.apply_to_box(track.box)

    def update(self, detections: list[Detection], frame_index: int) -> list[tuple[Track, Detection]]:
        """Advance the tracker by one frame. Returns matched pairs."""
        candidates: list[tuple[float, int, int]] = []
        for t_idx, track in enumerate(self.active):
            for d_idx, detection in enumerate(detections):
                score = self._score(track, detection)
                if score > 0:
                    candidates.append((score, t_idx, d_idx))

        candidates.sort(reverse=True)

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        matched: list[tuple[Track, Detection]] = []

        for score, t_idx, d_idx in candidates:
            if t_idx in used_tracks or d_idx in used_dets:
                continue
            used_tracks.add(t_idx)
            used_dets.add(d_idx)
            track = self.active[t_idx]
            track.update(detections[d_idx], frame_index)
            matched.append((track, detections[d_idx]))

        # Unmatched detections become new people.
        for d_idx, detection in enumerate(detections):
            if d_idx in used_dets:
                continue
            track = Track(
                track_id=self._next_id,
                box=detection.box,
                first_frame=frame_index,
                last_frame=frame_index,
                confidence_sum=detection.confidence,
                centroids=[detection.centroid],
            )
            self._next_id += 1
            self.active.append(track)
            matched.append((track, detection))

        # Unmatched tracks age out.
        still_active: list[Track] = []
        for t_idx, track in enumerate(self.active):
            if t_idx in used_tracks or track.last_frame == frame_index:
                still_active.append(track)
                continue
            track.missing += 1
            if track.missing > self.max_missing:
                self.closed.append(track)
            else:
                still_active.append(track)
        self.active = still_active

        return matched

    def finish(self) -> list[Track]:
        """Close all open tracks and return every track recorded."""
        self.closed.extend(self.active)
        self.active = []
        return self.closed

    def people(self, min_frames: int = TRACK_MIN_FRAMES) -> list[Track]:
        """Tracks long enough to be a person rather than detector flicker."""
        return [t for t in self.finish() if t.frames_seen >= min_frames]

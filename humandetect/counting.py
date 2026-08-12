"""Line-crossing counting — a virtual tripwire.

Why count crossings instead of unique people
--------------------------------------------
"How many unique people appeared" turned out to be unmeasurable with this
pipeline: it ranged from 95 to 483 on identical footage depending only on
tracker settings (docs/FINDINGS.md #2). The metric is exquisitely
sensitive to fragmentation, because every broken track invents a person.

Counting crossings of a line is far more robust. If one person is split
into three tracks, only the fragment that actually spans the line is
counted — the other two never cross it and contribute nothing. The metric
degrades gracefully where unique-person counting collapses.

This is also how real footfall systems work, and it maps directly onto the
question a retail or out-of-home advertising client actually asks: how
many people came through the door, and which way were they going.

Geometry
--------
A crossing is recorded when a track's centroid moves from one side of the
line to the other *and* the segment between its two positions genuinely
intersects the line segment. Testing the side alone is not enough: a
centroid can change side while passing well beyond the end of the line.

Direction comes from the sign of the side function, so entries and exits
are separable without the caller doing any geometry.
"""
from __future__ import annotations

from dataclasses import dataclass, field

Point = tuple[float, float]


def _side(point: Point, start: Point, end: Point) -> float:
    """Signed area: >0 one side of the line, <0 the other, 0 exactly on it."""
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Do segment p1-p2 and segment q1-q2 cross?"""
    d1 = _side(p1, q1, q2)
    d2 = _side(p2, q1, q2)
    d3 = _side(q1, p1, p2)
    d4 = _side(q2, p1, p2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


@dataclass
class Crossing:
    """One recorded crossing event."""

    track_id: int
    frame_index: int
    direction: str  # "in" or "out"
    point: Point


@dataclass
class CountingLine:
    """A virtual tripwire between two points in frame coordinates.

    `positive_is` names the direction taken when a track crosses from the
    negative side to the positive side, so the labels mean something to
    whoever reads the report.
    """

    start: Point
    end: Point
    positive_is: str = "in"

    @property
    def negative_is(self) -> str:
        return "out" if self.positive_is == "in" else "in"

    @classmethod
    def horizontal(cls, width: int, height: int, at: float = 0.5, positive_is: str = "in") -> "CountingLine":
        """A line straight across the frame at a fraction of its height."""
        y = height * at
        return cls((0.0, y), (float(width), y), positive_is)

    @classmethod
    def vertical(cls, width: int, height: int, at: float = 0.5, positive_is: str = "in") -> "CountingLine":
        """A line straight down the frame at a fraction of its width."""
        x = width * at
        return cls((x, 0.0), (x, float(height)), positive_is)

    @classmethod
    def parse(cls, spec: str, width: int, height: int, positive_is: str = "in") -> "CountingLine":
        """Build a line from a CLI string.

        Accepts:
          "horizontal"        across the middle
          "horizontal:0.7"    across at 70% of the height
          "vertical:0.3"      down at 30% of the width
          "x1,y1,x2,y2"       explicit pixel coordinates
        """
        spec = spec.strip().lower()

        if spec.startswith(("horizontal", "vertical")):
            kind, _, fraction = spec.partition(":")
            at = float(fraction) if fraction else 0.5
            if not 0.0 <= at <= 1.0:
                raise ValueError(f"Line position must be between 0 and 1, got {at}")
            builder = cls.horizontal if kind == "horizontal" else cls.vertical
            return builder(width, height, at, positive_is)

        parts = [p.strip() for p in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(
                f"Cannot parse line {spec!r}. Use 'horizontal', 'vertical:0.6', "
                "or four pixel coordinates 'x1,y1,x2,y2'."
            )
        x1, y1, x2, y2 = (float(p) for p in parts)
        return cls((x1, y1), (x2, y2), positive_is)

    def direction_of(self, previous: Point, current: Point) -> str | None:
        """Return "in"/"out" if this movement crossed the line, else None."""
        before = _side(previous, self.start, self.end)
        after = _side(current, self.start, self.end)

        if (before > 0) == (after > 0):
            return None  # same side, no crossing
        if not _segments_intersect(previous, current, self.start, self.end):
            return None  # changed side, but past the end of the line

        return self.positive_is if after > 0 else self.negative_is


@dataclass
class CrossingCounter:
    """Counts crossings of one line, at most one per track per direction change."""

    line: CountingLine
    crossings: list[Crossing] = field(default_factory=list)
    _last_point: dict[int, Point] = field(default_factory=dict)

    @property
    def total_in(self) -> int:
        return sum(1 for c in self.crossings if c.direction == "in")

    @property
    def total_out(self) -> int:
        return sum(1 for c in self.crossings if c.direction == "out")

    @property
    def total(self) -> int:
        return len(self.crossings)

    @property
    def net(self) -> int:
        """Positive means more went in than out — occupancy is rising."""
        return self.total_in - self.total_out

    def update(self, track_id: int, centroid: Point, frame_index: int) -> str | None:
        """Advance one track. Returns a direction if it crossed this frame."""
        previous = self._last_point.get(track_id)
        self._last_point[track_id] = centroid

        if previous is None:
            return None

        direction = self.line.direction_of(previous, centroid)
        if direction is None:
            return None

        self.crossings.append(
            Crossing(
                track_id=track_id,
                frame_index=frame_index,
                direction=direction,
                point=centroid,
            )
        )
        return direction

    def forget(self, track_id: int) -> None:
        """Drop state for a closed track."""
        self._last_point.pop(track_id, None)

    def summary(self, fps: float | None = None) -> str:
        lines = [
            f"Crossings   {self.total}",
            f"  in        {self.total_in}",
            f"  out       {self.total_out}",
            f"  net       {self.net:+d}",
        ]
        if fps and self.crossings:
            span = max(c.frame_index for c in self.crossings) / fps / 60 or 1
            lines.append(f"  rate      {self.total / span:.1f} crossings/minute")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "line": {"start": list(self.line.start), "end": list(self.line.end)},
            "total": self.total,
            "in": self.total_in,
            "out": self.total_out,
            "net": self.net,
            "events": [
                {
                    "track_id": c.track_id,
                    "frame": c.frame_index,
                    "direction": c.direction,
                }
                for c in self.crossings
            ],
        }

"""Audience metrics derived from tracks.

These are the numbers the original brief actually needed: not how many
boxes were drawn, but how many distinct people appeared, how long each
stayed, and when the scene was busiest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median

from .tracking import Track


@dataclass
class AudienceReport:
    """Summary of who appeared in a piece of footage."""

    frames: int
    fps: float
    unique_people: int
    dwell_seconds: list[float] = field(default_factory=list)
    peak_concurrent: int = 0
    peak_frame: int = 0
    occupancy: list[int] = field(default_factory=list)  # people present per frame
    transient_tracks: int = 0  # discarded as flicker

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.fps if self.fps else 0.0

    @property
    def mean_dwell(self) -> float:
        return mean(self.dwell_seconds) if self.dwell_seconds else 0.0

    @property
    def median_dwell(self) -> float:
        return median(self.dwell_seconds) if self.dwell_seconds else 0.0

    @property
    def mean_occupancy(self) -> float:
        return mean(self.occupancy) if self.occupancy else 0.0

    @property
    def footfall_per_minute(self) -> float:
        minutes = self.duration_seconds / 60.0
        return self.unique_people / minutes if minutes else 0.0

    def busiest_window(self, window_seconds: float = 10.0) -> tuple[float, float, float]:
        """Find the busiest stretch: (start_s, end_s, mean occupancy).

        This is the ad-placement signal — the moment with the most eyes on
        screen is where a break is worth the most.
        """
        if not self.occupancy or not self.fps:
            return (0.0, 0.0, 0.0)

        window = max(1, int(window_seconds * self.fps))
        if window >= len(self.occupancy):
            return (0.0, self.duration_seconds, self.mean_occupancy)

        running = sum(self.occupancy[:window])
        best_total, best_start = running, 0
        for i in range(window, len(self.occupancy)):
            running += self.occupancy[i] - self.occupancy[i - window]
            if running > best_total:
                best_total, best_start = running, i - window + 1

        return (
            best_start / self.fps,
            (best_start + window) / self.fps,
            best_total / window,
        )

    def summary(self) -> str:
        start, end, density = self.busiest_window()
        return "\n".join(
            [
                f"Footage            {self.duration_seconds / 60:.1f} min "
                f"({self.frames:,} frames @ {self.fps:.0f}fps)",
                "",
                f"Unique people      {self.unique_people}",
                f"Footfall           {self.footfall_per_minute:.1f} people/minute",
                f"Mean dwell         {self.mean_dwell:.1f}s",
                f"Median dwell       {self.median_dwell:.1f}s",
                "",
                f"Peak concurrent    {self.peak_concurrent} "
                f"(at {self.peak_frame / self.fps:.1f}s)",
                f"Mean concurrent    {self.mean_occupancy:.2f}",
                "",
                f"Busiest 10s window {start:.1f}s - {end:.1f}s "
                f"(mean {density:.1f} on screen)",
                f"Flicker discarded  {self.transient_tracks} short tracks",
            ]
        )

    def to_dict(self) -> dict:
        start, end, density = self.busiest_window()
        return {
            "frames": self.frames,
            "fps": self.fps,
            "duration_seconds": round(self.duration_seconds, 2),
            "unique_people": self.unique_people,
            "footfall_per_minute": round(self.footfall_per_minute, 2),
            "mean_dwell_seconds": round(self.mean_dwell, 2),
            "median_dwell_seconds": round(self.median_dwell, 2),
            "peak_concurrent": self.peak_concurrent,
            "peak_at_seconds": round(self.peak_frame / self.fps, 2) if self.fps else 0,
            "mean_concurrent": round(self.mean_occupancy, 2),
            "busiest_window": {
                "start_seconds": round(start, 2),
                "end_seconds": round(end, 2),
                "mean_on_screen": round(density, 2),
            },
            "transient_tracks_discarded": self.transient_tracks,
        }


def build_report(
    tracks: list[Track],
    all_tracks: list[Track],
    frames: int,
    fps: float,
) -> AudienceReport:
    """Turn finished tracks into an audience report.

    `tracks` are those long enough to count as people; `all_tracks`
    includes the short ones, so the report can say how much was discarded
    rather than hiding it.
    """
    occupancy = [0] * max(frames, 1)
    for track in tracks:
        start = max(0, track.first_frame)
        end = min(frames - 1, track.last_frame)
        for i in range(start, end + 1):
            occupancy[i] += 1

    peak_concurrent = max(occupancy) if occupancy else 0
    peak_frame = occupancy.index(peak_concurrent) if occupancy else 0

    return AudienceReport(
        frames=frames,
        fps=fps,
        unique_people=len(tracks),
        dwell_seconds=[t.frames_seen / fps for t in tracks] if fps else [],
        peak_concurrent=peak_concurrent,
        peak_frame=peak_frame,
        occupancy=occupancy,
        transient_tracks=len(all_tracks) - len(tracks),
    )

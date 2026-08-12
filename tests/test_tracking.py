"""Tests for tracking and audience metrics.

Synthetic detections are used so the expected identity assignments are
known exactly — a failure here means the tracker is wrong, not that the
detector behaved differently on a given frame.
"""
from __future__ import annotations

from humandetect.analytics import build_report
from humandetect.detector import Detection
from humandetect.tracking import PersonTracker


def det(box, conf=0.9) -> Detection:
    return Detection(class_id=1, label="person", confidence=conf, box=box)


def test_stationary_person_keeps_one_id():
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(10):
        tracker.update([det((100, 100, 50, 100))], frame)

    people = tracker.people(min_frames=1)

    assert len(people) == 1
    assert people[0].frames_seen == 10


def test_walking_person_keeps_one_id_while_boxes_overlap():
    """The core requirement: movement must not mint new identities."""
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(10):
        tracker.update([det((100 + frame * 5, 100, 50, 100))], frame)

    people = tracker.people(min_frames=1)

    assert len(people) == 1
    assert people[0].track_id == 1


def test_two_separate_people_get_separate_ids():
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(10):
        tracker.update([det((50, 100, 40, 90)), det((600, 100, 40, 90))], frame)

    people = tracker.people(min_frames=1)

    assert len(people) == 2
    assert {p.track_id for p in people} == {1, 2}


def test_brief_occlusion_does_not_split_a_person():
    """Someone walking behind a sign for a few frames stays one person."""
    tracker = PersonTracker(frame_diagonal=1000, max_missing=15)
    for frame in range(5):
        tracker.update([det((100, 100, 50, 100))], frame)
    for frame in range(5, 10):  # occluded
        tracker.update([], frame)
    for frame in range(10, 15):
        tracker.update([det((100, 100, 50, 100))], frame)

    people = tracker.people(min_frames=1)

    assert len(people) == 1


def test_long_absence_closes_the_track():
    """Beyond the tolerance, a reappearance is treated as a new person."""
    tracker = PersonTracker(frame_diagonal=1000, max_missing=3)
    for frame in range(5):
        tracker.update([det((100, 100, 50, 100))], frame)
    for frame in range(5, 20):
        tracker.update([], frame)
    for frame in range(20, 25):
        tracker.update([det((100, 100, 50, 100))], frame)

    people = tracker.people(min_frames=1)

    assert len(people) == 2


def test_flicker_is_discarded_by_min_frames():
    """A one-frame false positive must not inflate the headcount."""
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(20):
        boxes = [det((100, 100, 50, 100))]
        if frame == 7:
            boxes.append(det((700, 300, 30, 40), conf=0.51))
        tracker.update(boxes, frame)

    assert len(tracker.people(min_frames=5)) == 1


def test_centroid_fallback_links_a_fast_mover():
    """Non-overlapping boxes across frames still belong to one person."""
    tracker = PersonTracker(frame_diagonal=1000, iou_threshold=0.9)
    tracker.update([det((100, 100, 40, 80))], 0)
    tracker.update([det((150, 100, 40, 80))], 1)  # no overlap at IoU 0.9

    assert len(tracker.people(min_frames=1)) == 1


def test_report_counts_unique_people_not_detections():
    """The headline distinction: 3 people over 100 frames is 3, not 300."""
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(100):
        tracker.update(
            [det((50, 100, 40, 90)), det((400, 100, 40, 90)), det((750, 100, 40, 90))],
            frame,
        )

    all_tracks = list(tracker.finish())
    people = [t for t in all_tracks if t.frames_seen >= 5]
    report = build_report(people, all_tracks, frames=100, fps=25.0)

    assert report.unique_people == 3
    assert report.peak_concurrent == 3
    assert report.mean_dwell == 4.0  # 100 frames / 25 fps


def test_busiest_window_finds_the_crowded_stretch():
    tracker = PersonTracker(frame_diagonal=1000)
    for frame in range(100):
        boxes = [det((50, 100, 40, 90))]
        if 60 <= frame < 90:  # a crowd arrives late
            boxes += [det((300, 100, 40, 90)), det((550, 100, 40, 90))]
        tracker.update(boxes, frame)

    all_tracks = list(tracker.finish())
    people = [t for t in all_tracks if t.frames_seen >= 5]
    report = build_report(people, all_tracks, frames=100, fps=10.0)

    start, end, density = report.busiest_window(window_seconds=3.0)

    assert start >= 5.0  # the busy stretch begins at frame 60 == 6.0s
    assert density > 1.0


def test_empty_footage_produces_zeroed_report():
    report = build_report([], [], frames=0, fps=25.0)

    assert report.unique_people == 0
    assert report.mean_dwell == 0.0
    assert report.busiest_window() == (0.0, 0.0, 0.0)

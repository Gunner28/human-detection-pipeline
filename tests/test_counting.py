"""Tests for line-crossing counting.

The geometry is the whole point of this module, so it is tested directly
rather than through the pipeline. In particular: changing side is not the
same as crossing, and the difference matters at the ends of the line.
"""
from __future__ import annotations

import pytest

from humandetect.counting import CountingLine, CrossingCounter


def horizontal_line() -> CountingLine:
    """A line across a 100x100 frame at y=50."""
    return CountingLine(start=(0.0, 50.0), end=(100.0, 50.0), positive_is="in")


def test_crossing_downward_is_counted():
    line = horizontal_line()

    assert line.direction_of((50.0, 40.0), (50.0, 60.0)) is not None


def test_crossing_the_other_way_reverses_direction():
    line = horizontal_line()

    down = line.direction_of((50.0, 40.0), (50.0, 60.0))
    up = line.direction_of((50.0, 60.0), (50.0, 40.0))

    assert down != up
    assert {down, up} == {"in", "out"}


def test_movement_on_one_side_is_not_a_crossing():
    line = horizontal_line()

    assert line.direction_of((10.0, 10.0), (90.0, 40.0)) is None


def test_changing_side_beyond_the_end_of_the_line_is_not_a_crossing():
    """The critical case: side alone is not enough.

    At x=200 the track is well past the line's end at x=100, so although
    it moves from above y=50 to below it, nothing was actually crossed.
    """
    line = horizontal_line()

    assert line.direction_of((200.0, 40.0), (200.0, 60.0)) is None


def test_counter_records_one_event_per_crossing():
    counter = CrossingCounter(line=horizontal_line())

    counter.update(track_id=1, centroid=(50.0, 40.0), frame_index=0)
    counter.update(track_id=1, centroid=(50.0, 45.0), frame_index=1)
    counter.update(track_id=1, centroid=(50.0, 60.0), frame_index=2)

    assert counter.total == 1


def test_first_sighting_never_counts():
    """A track appearing already past the line has not crossed it."""
    counter = CrossingCounter(line=horizontal_line())

    counter.update(track_id=1, centroid=(50.0, 90.0), frame_index=0)

    assert counter.total == 0


def test_walking_back_and_forth_counts_both_ways():
    counter = CrossingCounter(line=horizontal_line())

    for frame, y in enumerate([40.0, 60.0, 40.0]):
        counter.update(track_id=1, centroid=(50.0, y), frame_index=frame)

    assert counter.total == 2
    assert counter.total_in == 1
    assert counter.total_out == 1
    assert counter.net == 0


def test_separate_tracks_are_counted_separately():
    counter = CrossingCounter(line=horizontal_line())

    for track_id in (1, 2, 3):
        counter.update(track_id, (50.0, 40.0), 0)
        counter.update(track_id, (50.0, 60.0), 1)

    assert counter.total == 3


def test_fragmentation_does_not_inflate_the_count():
    """The reason this metric exists.

    One person split into three tracks: only the fragment that actually
    spans the line contributes. Unique-person counting would report three.
    """
    counter = CrossingCounter(line=horizontal_line())

    # Fragment A approaches but never reaches the line.
    counter.update(1, (50.0, 20.0), 0)
    counter.update(1, (50.0, 35.0), 1)
    # Fragment B spans it.
    counter.update(2, (50.0, 45.0), 2)
    counter.update(2, (50.0, 55.0), 3)
    # Fragment C continues away on the far side.
    counter.update(3, (50.0, 70.0), 4)
    counter.update(3, (50.0, 90.0), 5)

    assert counter.total == 1


def test_horizontal_helper_places_the_line_by_fraction():
    line = CountingLine.horizontal(width=200, height=100, at=0.25)

    assert line.start == (0.0, 25.0)
    assert line.end == (200.0, 25.0)


def test_vertical_helper_places_the_line_by_fraction():
    line = CountingLine.vertical(width=200, height=100, at=0.5)

    assert line.start == (100.0, 0.0)
    assert line.end == (100.0, 100.0)


@pytest.mark.parametrize(
    "spec,expected_start,expected_end",
    [
        ("horizontal", (0.0, 50.0), (100.0, 50.0)),
        ("horizontal:0.2", (0.0, 20.0), (100.0, 20.0)),
        ("vertical:0.5", (50.0, 0.0), (50.0, 100.0)),
        ("10,20,30,40", (10.0, 20.0), (30.0, 40.0)),
    ],
)
def test_parse_accepts_each_supported_form(spec, expected_start, expected_end):
    line = CountingLine.parse(spec, width=100, height=100)

    assert line.start == expected_start
    assert line.end == expected_end


def test_parse_rejects_nonsense_with_a_useful_message():
    with pytest.raises(ValueError, match="Cannot parse line"):
        CountingLine.parse("diagonal-ish", width=100, height=100)


def test_parse_rejects_an_out_of_range_fraction():
    with pytest.raises(ValueError, match="between 0 and 1"):
        CountingLine.parse("horizontal:1.8", width=100, height=100)


def test_forget_drops_track_state():
    counter = CrossingCounter(line=horizontal_line())
    counter.update(1, (50.0, 40.0), 0)
    counter.forget(1)

    # With no remembered position, the next sighting is treated as a first.
    counter.update(1, (50.0, 60.0), 1)

    assert counter.total == 0

"""Tests for duplicate-box suppression.

These pin the defect this module exists to fix. OpenCV's
`dnn_DetectionModel.detect(nmsThreshold=...)` does not suppress anything
with this model — sweeping it from 0.6 to 0.1 on a single-person image
returned an identical five detections every time. The real boxes from that
image are used below as a regression fixture.
"""
from __future__ import annotations

from humandetect.detector import Detection, _iou_and_containment, suppress


def det(box, conf, class_id=1) -> Detection:
    return Detection(class_id=class_id, label="person", confidence=conf, box=box)


# Boxes measured from samples/manbenz.png — one man, one car, but the model
# emits three overlapping "person" boxes.
MANBENZ_PEOPLE = [
    det((81, 106, 82, 95), 0.748),
    det((78, 84, 126, 112), 0.517),
    det((69, 70, 182, 111), 0.505),
]


def test_iou_matches_hand_computed_value():
    iou, _ = _iou_and_containment((0, 0, 10, 10), (0, 0, 10, 10))
    assert iou == 1.0


def test_disjoint_boxes_score_zero():
    assert _iou_and_containment((0, 0, 10, 10), (50, 50, 10, 10)) == (0.0, 0.0)


def test_containment_catches_nesting_that_iou_misses():
    """A small box fully inside a large one: containment 1.0, low IoU."""
    outer = (0, 0, 100, 100)
    inner = (10, 10, 20, 20)

    iou, containment = _iou_and_containment(outer, inner)

    assert containment == 1.0
    assert iou < 0.05  # IoU alone would never suppress this


def test_three_overlapping_people_collapse_to_one():
    """The regression case: one man reported three times."""
    kept = suppress(MANBENZ_PEOPLE)

    assert len(kept) == 1
    # The highest-confidence box survives.
    assert kept[0].confidence == 0.748


def test_suppression_keeps_the_highest_confidence_box():
    kept = suppress([det((0, 0, 100, 100), 0.6), det((5, 5, 100, 100), 0.9)])

    assert len(kept) == 1
    assert kept[0].confidence == 0.9


def test_different_classes_are_never_suppressed_against_each_other():
    """A person standing in front of a car must not delete the car."""
    kept = suppress(
        [det((0, 0, 100, 100), 0.9, class_id=1), det((0, 0, 100, 100), 0.8, class_id=3)]
    )

    assert len(kept) == 2


def test_genuinely_separate_people_are_both_kept():
    kept = suppress([det((0, 0, 50, 100), 0.9), det((200, 0, 50, 100), 0.8)])

    assert len(kept) == 2


def test_empty_input_returns_empty():
    assert suppress([]) == []


def test_detection_helpers():
    d = det((10, 20, 30, 40), 0.9)

    assert d.is_person is True
    assert d.area == 1200
    assert d.centroid == (25.0, 40.0)

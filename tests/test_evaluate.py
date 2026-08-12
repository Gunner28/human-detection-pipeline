"""Tests for the evaluation harness.

Synthetic boxes are used so the expected precision/recall is known by
construction — a failure means the metric is wrong, not that the detector
behaved differently today.
"""
from __future__ import annotations

from humandetect.detector import Detection
from humandetect.evaluate import BoxReport, CountReport, match_frame


def pred(box, conf=0.9) -> Detection:
    return Detection(class_id=1, label="person", confidence=conf, box=box)


def test_perfect_match_scores_one():
    result = match_frame([pred((10, 10, 40, 100))], [(10, 10, 40, 100)], min_height=0)

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == 1.0
    assert result.recall == 1.0


def test_spurious_box_is_a_false_positive():
    result = match_frame(
        [pred((10, 10, 40, 100)), pred((500, 10, 40, 100))],
        [(10, 10, 40, 100)],
        min_height=0,
    )

    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.precision == 0.5


def test_missed_person_is_a_false_negative():
    result = match_frame([], [(10, 10, 40, 100)], min_height=0)

    assert result.false_negatives == 1
    assert result.recall == 0.0


def test_box_below_iou_threshold_does_not_match():
    """A box in roughly the right area is not automatically correct."""
    result = match_frame(
        [pred((0, 0, 40, 100))], [(35, 0, 40, 100)], iou_threshold=0.5, min_height=0
    )

    assert result.true_positives == 0
    assert result.false_positives == 1
    assert result.false_negatives == 1


def test_one_prediction_cannot_claim_two_truths():
    result = match_frame(
        [pred((10, 10, 40, 100))],
        [(10, 10, 40, 100), (12, 12, 40, 100)],
        min_height=0,
    )

    assert result.true_positives == 1
    assert result.false_negatives == 1


def test_min_height_excludes_small_people_from_both_sides():
    """Distant figures are excluded from ground truth and predictions
    alike, so they are not silently scored as detector errors."""
    result = match_frame(
        [pred((10, 10, 20, 30)), pred((100, 10, 40, 100))],
        [(10, 10, 20, 30), (100, 10, 40, 100)],
        min_height=60,
    )

    assert result.true_positives == 1
    assert result.false_positives == 0
    assert result.false_negatives == 0


def test_box_report_aggregates_across_frames():
    report = BoxReport(iou_threshold=0.5, min_height=60)
    report.frames.append(match_frame([pred((0, 0, 40, 100))], [(0, 0, 40, 100)], min_height=0))
    report.frames.append(match_frame([], [(0, 0, 40, 100)], min_height=0))

    assert report.true_positives == 1
    assert report.false_negatives == 1
    assert report.recall == 0.5
    assert report.precision == 1.0
    assert abs(report.f1 - 2 / 3) < 1e-9


def test_count_report_bias_is_signed():
    """Positive bias means over-counting; the sign carries the meaning."""
    over = CountReport(min_height=60, confidence=0.5, predicted=[8], actual=[5])
    under = CountReport(min_height=60, confidence=0.5, predicted=[3], actual=[5])

    assert over.bias == 3.0
    assert under.bias == -2.0


def test_count_report_mae_ignores_sign():
    """Over- and under-counting must not cancel out."""
    report = CountReport(min_height=60, confidence=0.5, predicted=[8, 1], actual=[5, 5])

    assert report.bias == -0.5  # +3 and -4 average out, hiding the problem
    assert report.mean_absolute_error == 3.5  # MAE exposes it


def test_empty_count_report_does_not_divide_by_zero():
    report = CountReport(min_height=60, confidence=0.5)

    assert report.mean_absolute_error == 0.0
    assert report.bias == 0.0
    assert report.ratio == 0.0

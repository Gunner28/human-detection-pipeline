"""Tests for camera-motion compensation.

Synthetic frames are used so the true camera motion is known exactly: a
fixed image is shifted by a known number of pixels, and the estimator must
recover that shift.
"""
from __future__ import annotations

import cv2
import numpy as np

from humandetect.detector import Detection
from humandetect.motion import CameraMotionEstimator, Transform


def textured_frame(width: int = 320, height: int = 240, seed: int = 0) -> np.ndarray:
    """A frame with enough texture for corner detection to work."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    # Add hard structure — pure noise gives unstable corners.
    for i in range(6):
        cv2.rectangle(
            frame,
            (20 + i * 45, 30 + i * 20),
            (60 + i * 45, 90 + i * 20),
            (255, 255, 255),
            -1,
        )
    return frame


def shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame, matrix, (frame.shape[1], frame.shape[0]))


def test_identity_transform_leaves_a_box_untouched():
    box = (10, 20, 30, 40)

    assert Transform(None).apply_to_box(box) == box
    assert Transform(None).magnitude == 0.0


def test_first_frame_yields_no_transform():
    """There is nothing to compare against yet."""
    estimator = CameraMotionEstimator()

    transform = estimator.update(textured_frame())

    assert transform.is_identity


def test_estimator_recovers_a_known_translation():
    estimator = CameraMotionEstimator()
    frame = textured_frame()

    estimator.update(frame)
    transform = estimator.update(shift(frame, 12, -8))

    assert not transform.is_identity
    dx, dy = transform.translation
    assert abs(dx - 12) < 3.0
    assert abs(dy - (-8)) < 3.0


def test_a_static_camera_reports_near_zero_motion():
    estimator = CameraMotionEstimator()
    frame = textured_frame()

    estimator.update(frame)
    transform = estimator.update(frame.copy())

    assert transform.magnitude < 1.0


def test_disabled_estimator_never_reports_motion():
    estimator = CameraMotionEstimator(enabled=False)
    frame = textured_frame()

    estimator.update(frame)
    transform = estimator.update(shift(frame, 20, 20))

    assert transform.is_identity


def test_applying_a_transform_moves_a_box_by_the_camera_motion():
    estimator = CameraMotionEstimator()
    frame = textured_frame()
    estimator.update(frame)
    transform = estimator.update(shift(frame, 10, 0))

    moved = transform.apply_to_box((100, 100, 40, 80))

    assert abs(moved[0] - 110) < 4
    assert abs(moved[1] - 100) < 4
    # Size is preserved under pure translation.
    assert abs(moved[2] - 40) < 3
    assert abs(moved[3] - 80) < 3


def test_people_are_masked_out_of_the_motion_estimate():
    """Features on people would bias the estimate toward crowd motion."""
    estimator = CameraMotionEstimator()
    frame = textured_frame()
    covering_everything = [
        Detection(class_id=1, label="person", confidence=0.9, box=(0, 0, 320, 240))
    ]

    estimator.update(frame, covering_everything)
    transform = estimator.update(shift(frame, 10, 0), covering_everything)

    # With every candidate point masked out, no estimate is possible —
    # the estimator must degrade to identity rather than invent motion.
    assert transform.is_identity

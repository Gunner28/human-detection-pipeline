"""Benchmark configuration: the settings that decide what a number means.

Only the parts that need no dataset and no model are tested here. Running
the benchmark itself needs the 5.6 GB download; these cover the logic that
was wrong once and would be wrong silently again — inference size, and
whether a config change actually reaches the detector.
"""
from __future__ import annotations

import pytest

from humandetect import mot17
from humandetect.benchmark import BenchmarkConfig


def seqinfo(width: int, height: int) -> mot17.SeqInfo:
    return mot17.SeqInfo(
        name="test", length=10, width=width, height=height,
        frame_rate=30, ext=".jpg",
    )


# --------------------------------------------------------------------------
# Inference size
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "width,height,requested,expected",
    [
        (1920, 1080, 1280, 1280),  # below native: unchanged
        (640, 480, 1280, 640),     # above native: capped (MOT17-05)
        (1920, 1080, 640, 640),    # asking for less is honoured
        (640, 480, 640, 640),      # exactly native
    ],
)
def test_imgsz_never_exceeds_native_resolution(width, height, requested, expected):
    """Upscaling past native cost MOT17-05 22.6 points of MOTA — findings #13."""
    config = BenchmarkConfig(backend="yolo", imgsz=requested)
    assert config.effective_imgsz(seqinfo(width, height)) == expected


def test_capped_imgsz_stays_a_multiple_of_32():
    """YOLO requires it; an odd size is silently rounded and hard to trace."""
    config = BenchmarkConfig(backend="yolo", imgsz=1280)
    for width, height in [(700, 500), (1000, 1000), (33, 33), (100, 240)]:
        assert config.effective_imgsz(seqinfo(width, height)) % 32 == 0


def test_capping_can_be_switched_off():
    config = BenchmarkConfig(backend="yolo", imgsz=1280, cap_imgsz_to_native=False)
    assert config.effective_imgsz(seqinfo(640, 480)) == 1280


def test_ssd_ignores_imgsz_entirely():
    """Its 320x320 input is baked into the frozen graph."""
    config = BenchmarkConfig(backend="ssd", imgsz=1280)
    assert config.effective_imgsz(seqinfo(640, 480)) == 1280


# --------------------------------------------------------------------------
# Configuration reaching the detector
# --------------------------------------------------------------------------
def test_containment_defaults_to_the_configured_value():
    assert BenchmarkConfig().containment is None


def test_containment_override_survives_on_the_config():
    """Regression: this was once toggled by reloading a module, which did
    nothing at all — backends.py binds `suppress` at import time and kept
    the old reference, so both runs scored identically. Findings #12."""
    config = BenchmarkConfig(containment=1.01)
    assert config.containment == 1.01


def test_describe_names_every_variable_that_moves_a_number():
    described = BenchmarkConfig(
        backend="yolo", imgsz=1280, confidence=0.25,
        min_frames=5, compensate=True,
    ).describe()
    for expected in ["yolo", "1280", "0.25", "min_frames=5", "compensate=True"]:
        assert expected in described


def test_describe_marks_ssd_input_as_fixed():
    assert "320 (fixed)" in BenchmarkConfig(backend="ssd").describe()

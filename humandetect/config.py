"""Tunables for the detection and tracking pipeline.

Everything configurable lives here and reads an environment variable, so
behaviour can be changed without editing code.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# --- Model ---------------------------------------------------------------
FROZEN_GRAPH = Path(os.environ.get("HD_FROZEN_GRAPH", MODELS_DIR / "frozen_inference_graph.pb"))
CONFIG_FILE = Path(
    os.environ.get("HD_CONFIG_FILE", MODELS_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
)
LABELS_FILE = Path(os.environ.get("HD_LABELS_FILE", MODELS_DIR / "coco_labels.txt"))

# SSD MobileNet v3 was trained at 320x320 with inputs scaled to [-1, 1].
INPUT_SIZE = (320, 320)
INPUT_SCALE = 1.0 / 127.5
INPUT_MEAN = (127.5, 127.5, 127.5)

PERSON_CLASS_ID = 1  # COCO, as emitted by this model (1-indexed)

# --- Detection -----------------------------------------------------------
CONFIDENCE_THRESHOLD = float(os.environ.get("HD_CONFIDENCE", "0.5"))

# Duplicate suppression. OpenCV's own nmsThreshold is a no-op with this
# model, so these drive the suppression implemented in detector.suppress().
IOU_THRESHOLD = float(os.environ.get("HD_IOU", "0.45"))
CONTAINMENT_THRESHOLD = float(os.environ.get("HD_CONTAINMENT", "0.7"))

# --- Tracking ------------------------------------------------------------
# Minimum IoU for a detection to continue an existing track.
TRACK_IOU_THRESHOLD = float(os.environ.get("HD_TRACK_IOU", "0.3"))

# Fallback when boxes do not overlap at all (fast motion, low frame rate):
# match on centroid distance, as a fraction of the frame diagonal.
TRACK_DISTANCE_FRACTION = float(os.environ.get("HD_TRACK_DISTANCE", "0.08"))

# Frames a track may go unmatched before it is closed. At 30fps, 15 frames
# is half a second — long enough to survive a brief occlusion (someone
# walking behind a sign) without merging two different people.
TRACK_MAX_MISSING = int(os.environ.get("HD_TRACK_MAX_MISSING", "15"))

# Tracks shorter than this are treated as detector flicker, not people.
TRACK_MIN_FRAMES = int(os.environ.get("HD_TRACK_MIN_FRAMES", "5"))

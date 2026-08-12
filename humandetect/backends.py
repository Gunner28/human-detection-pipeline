"""Pluggable detector backends behind one interface.

The point of this module is comparability. Both detectors return the same
`Detection` objects, go through the same duplicate suppression, and are
measured by the same harness — so a difference in the numbers is a
difference in the model, not in the plumbing around it.

Backends
--------
`ssd`   SSD MobileNet v3 (COCO) through OpenCV's DNN module. The original
        Comviva pipeline. No deep-learning runtime required.

`yolo`  YOLOv8 through ultralytics. Modern anchor-free detector with a
        much stronger backbone.

Licensing note: ultralytics is AGPL-3.0. That is fine for a public
portfolio repository, but it would need replacing (or a commercial
licence) before shipping YOLOv8 inside closed-source software. The SSD
path has no such constraint, which is a real reason to keep both.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import CONFIDENCE_THRESHOLD, PERSON_CLASS_ID
from .detector import Detection, PersonDetector, suppress


class DetectorBackend(Protocol):
    """What every backend must provide."""

    name: str

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        people_only: bool = True,
        deduplicate: bool = True,
    ) -> list[Detection]: ...


class SSDBackend:
    """SSD MobileNet v3 via OpenCV DNN — the original pipeline."""

    name = "ssd-mobilenet-v3"

    def __init__(self) -> None:
        self._detector = PersonDetector()

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        people_only: bool = True,
        deduplicate: bool = True,
    ) -> list[Detection]:
        return self._detector.detect(
            frame,
            confidence_threshold=confidence_threshold,
            people_only=people_only,
            deduplicate=deduplicate,
        )


class YOLOBackend:
    """YOLOv8 via ultralytics.

    YOLO applies its own non-maximum suppression internally and, unlike
    OpenCV's `dnn_DetectionModel`, that suppression actually works. The
    project's own `suppress()` still runs when `deduplicate=True` so both
    backends are treated identically by the harness; on YOLO output it
    almost never removes anything, which is itself a useful signal.
    """

    name = "yolov8n"

    def __init__(self, weights: str = "yolov8n.pt") -> None:
        from ultralytics import YOLO  # imported lazily: heavy, and optional

        self.model = YOLO(weights)
        self.name = weights.replace(".pt", "")

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        people_only: bool = True,
        deduplicate: bool = True,
    ) -> list[Detection]:
        results = self.model.predict(
            frame,
            conf=confidence_threshold,
            verbose=False,
        )

        detections: list[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                # ultralytics is 0-indexed (person == 0); this project's
                # Detection uses the model's 1-indexed COCO convention, so
                # shift to keep PERSON_CLASS_ID meaningful across backends.
                raw_class = int(box.cls.item())
                class_id = raw_class + 1
                if people_only and class_id != PERSON_CLASS_ID:
                    continue

                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                detections.append(
                    Detection(
                        class_id=class_id,
                        label=names.get(raw_class, f"class_{raw_class}"),
                        confidence=float(box.conf.item()),
                        box=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                    )
                )

        return suppress(detections) if deduplicate else detections


def load_backend(name: str) -> DetectorBackend:
    """Build a backend by name: 'ssd' or 'yolo' (optionally 'yolo:yolov8s.pt')."""
    if name == "ssd":
        return SSDBackend()
    if name.startswith("yolo"):
        _, _, weights = name.partition(":")
        return YOLOBackend(weights or "yolov8n.pt")
    raise ValueError(f"Unknown backend: {name!r}. Use 'ssd' or 'yolo'.")

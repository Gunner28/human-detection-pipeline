"""SSD MobileNet v3 person detection via OpenCV's DNN module.

This is the detection core from the original Comviva notebook, corrected and
made runnable outside a notebook. The model is a TensorFlow frozen graph
(SSD MobileNet v3 Large, trained on COCO) loaded through cv2.dnn — no
TensorFlow runtime required, which is why it runs anywhere OpenCV does.

COCO class 1 is `person`. Everything here filters to that by default; the
full 80-class output is still available when you want scene context.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import (
    CONFIDENCE_THRESHOLD,
    CONFIG_FILE,
    CONTAINMENT_THRESHOLD,
    FROZEN_GRAPH,
    INPUT_MEAN,
    INPUT_SCALE,
    INPUT_SIZE,
    IOU_THRESHOLD,
    LABELS_FILE,
    PERSON_CLASS_ID,
)


@dataclass(frozen=True)
class Detection:
    """One detected object in one frame."""

    class_id: int
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, w, h

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    @property
    def area(self) -> int:
        return self.box[2] * self.box[3]

    @property
    def centroid(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


def _iou_and_containment(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[float, float]:
    """Return (IoU, containment) for two x/y/w/h boxes.

    Containment is intersection over the *smaller* box's area. It catches
    the nested-duplicate case that IoU misses: a small box almost entirely
    inside a large one can have high containment but low IoU.
    """
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0, 0.0

    area_a, area_b = aw * ah, bw * bh
    union = area_a + area_b - intersection
    return intersection / union, intersection / min(area_a, area_b)


def suppress(
    detections: list["Detection"],
    iou_threshold: float = IOU_THRESHOLD,
    containment_threshold: float = CONTAINMENT_THRESHOLD,
) -> list["Detection"]:
    """Greedy non-maximum suppression, per class.

    This exists because OpenCV's `dnn_DetectionModel.detect(nmsThreshold=...)`
    is a no-op with this model — verified by sweeping the threshold from 0.6
    down to 0.1 and getting an identical five detections every time, on an
    image containing one person and one car. Without this, the same subject
    is counted several times and every downstream count is inflated.
    """
    kept: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.confidence, reverse=True):
        duplicate = False
        for chosen in kept:
            if chosen.class_id != det.class_id:
                continue
            iou, containment = _iou_and_containment(chosen.box, det.box)
            if iou >= iou_threshold or containment >= containment_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


WEIGHTS_URL = (
    "http://download.tensorflow.org/models/object_detection/"
    "ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
)


def ensure_weights(target: Path = FROZEN_GRAPH) -> None:
    """Download the frozen graph on first use.

    The weights are 13 MB, so they are not committed. Fetching them
    automatically means someone can clone the repository, point the CLI at
    their own video, and have it work without reading the README first.
    """
    if target.is_file():
        return

    import tarfile
    import tempfile
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading detection weights (~13 MB) to {target}…", flush=True)

    member = "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "weights.tar.gz"
            urllib.request.urlretrieve(WEIGHTS_URL, archive)  # noqa: S310
            with tarfile.open(archive) as tar:
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"{member} missing from the archive")
                target.write_bytes(extracted.read())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not download the model weights: {exc}\n"
            f"Fetch them manually from {WEIGHTS_URL} and place "
            f"frozen_inference_graph.pb at {target}."
        ) from exc

    print("Weights ready.", flush=True)


def load_labels(path: Path = LABELS_FILE) -> list[str]:
    """Read COCO class names, one per line.

    The original notebook read `labels.txt` while the file on disk was named
    `lable.txt`, so it raised FileNotFoundError as written.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"Label file not found: {path}\n"
            "Expected one class name per line (COCO order, 'person' first)."
        )
    text = path.read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


class PersonDetector:
    """Loads the frozen graph once and detects on frames."""

    def __init__(
        self,
        frozen_graph: Path = FROZEN_GRAPH,
        config_file: Path = CONFIG_FILE,
        labels_file: Path = LABELS_FILE,
    ) -> None:
        if not frozen_graph.is_file():
            ensure_weights(frozen_graph)

        for required in (frozen_graph, config_file):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Model file not found: {required}\n"
                    "Run scripts/download_models.sh to fetch the weights."
                )

        self.labels = load_labels(labels_file)
        self.model = cv2.dnn_DetectionModel(str(frozen_graph), str(config_file))
        self.model.setInputSize(*INPUT_SIZE)
        self.model.setInputScale(INPUT_SCALE)
        self.model.setInputMean(INPUT_MEAN)
        self.model.setInputSwapRB(True)

    def label_for(self, class_id: int) -> str:
        """Map a 1-indexed class id to its name, safely."""
        index = class_id - 1
        if 0 <= index < len(self.labels):
            return self.labels[index]
        return f"class_{class_id}"

    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        people_only: bool = True,
        deduplicate: bool = True,
        containment_threshold: float | None = None,
    ) -> list[Detection]:
        """Detect objects in a single frame.

        Duplicate boxes are removed by `suppress()` rather than by OpenCV's
        own `nmsThreshold`, which does nothing here (see that function).
        Pass `deduplicate=False` to see the raw model output.

        `containment_threshold` overrides the configured value for this call
        only. It is a parameter rather than a module constant because
        experiments need to vary it: reassigning the constant and reloading
        the module does not reach callers that already imported `suppress`,
        so the toggle silently does nothing and the run looks like evidence
        of no effect.
        """
        class_ids, confidences, boxes = self.model.detect(
            frame,
            confThreshold=confidence_threshold,
        )

        # OpenCV returns empty tuples (not arrays) when nothing is found.
        if len(class_ids) == 0:
            return []

        detections: list[Detection] = []
        for class_id, confidence, box in zip(
            np.array(class_ids).flatten(),
            np.array(confidences).flatten(),
            boxes,
        ):
            class_id = int(class_id)
            if people_only and class_id != PERSON_CLASS_ID:
                continue
            x, y, w, h = (int(v) for v in box)
            detections.append(
                Detection(
                    class_id=class_id,
                    label=self.label_for(class_id),
                    confidence=float(confidence),
                    box=(x, y, w, h),
                )
            )
        if not deduplicate:
            return detections
        if containment_threshold is None:
            return suppress(detections)
        return suppress(detections, containment_threshold=containment_threshold)


def draw(
    frame: np.ndarray,
    detections: list[Detection],
    colour: tuple[int, int, int] = (0, 200, 255),
) -> np.ndarray:
    """Annotate a copy of the frame with boxes and labels."""
    annotated = frame.copy()
    for det in detections:
        x, y, w, h = det.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), colour, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            annotated,
            caption,
            (x, max(y - 8, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
            cv2.LINE_AA,
        )
    return annotated

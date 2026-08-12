# Human Detection Pipeline

Deep-learning human detection and tracking in video — detect people frame
by frame, follow each person across frames, and report where and when they
appeared.

Built on a convolutional neural network (SSD MobileNet v3, trained on
COCO), originally developed during an internship at Comviva
(Oct 2019 – Oct 2022) for audience analysis in advertising, and since
extended with tracking, evaluation and a second detector backend.

```bash
pip install -r requirements.txt
python cli.py video my_video.mp4 -o annotated.mp4
```

Model weights download automatically on first run.

---

## The model

**SSD MobileNet v3 Large** — a single-shot detector built on a MobileNetV3
convolutional backbone, trained on the COCO dataset (80 object classes,
`person` among them).

| | |
|---|---|
| Architecture | Single Shot MultiBox Detector (SSD) |
| Backbone | MobileNetV3-Large, depthwise-separable convolutions |
| Training set | COCO — 80 classes, ~200k labelled images |
| Input | 320×320 RGB, normalised to [-1, 1] |
| Output | Class scores and bounding boxes per anchor |
| Inference | OpenCV DNN module, ~30 ms/frame on CPU |

Inference runs through OpenCV's DNN module, which reads the TensorFlow
frozen graph directly. In practice that means the trained network deploys
without installing TensorFlow or PyTorch — useful for edge and embedded
targets where a full training framework will not fit.

**YOLOv8** is available as a second backend
([`humandetect/backends.py`](humandetect/backends.py)), so detectors can be
swapped behind one interface and compared on identical footage.

```bash
pip install ultralytics    # then use the yolo backend
```

---

## Pipeline

```
video ─► CNN detection ─► duplicate suppression ─► tracking ─► analytics
         (SSD/YOLOv8)      (IoU + containment)     (IDs across   (presence,
                                                    frames)       dwell, flow)
```

| Module | What it does |
|---|---|
| [`detector.py`](humandetect/detector.py) | CNN inference, person filtering, duplicate suppression |
| [`backends.py`](humandetect/backends.py) | SSD MobileNet / YOLOv8 behind one interface |
| [`tracking.py`](humandetect/tracking.py) | Assigns and maintains a stable identity per person |
| [`motion.py`](humandetect/motion.py) | Camera-motion compensation via optical flow |
| [`analytics.py`](humandetect/analytics.py) | Dwell time, occupancy, busiest window |
| [`counting.py`](humandetect/counting.py) | Directional line-crossing counts |
| [`evaluate.py`](humandetect/evaluate.py) | Precision, recall, F1 against labelled frames |
| [`config.py`](humandetect/config.py) | Every parameter, environment-overridable |

---

## Usage

```bash
python cli.py image  photo.jpg                 # detect people in a still
python cli.py video  clip.mp4 -o out.mp4       # annotate every frame
python cli.py webcam                           # live from a camera
python cli.py analyse clip.mp4                 # tracking + presence metrics
python cli.py count  clip.mp4 -l horizontal    # directional flow across a line
```

Every parameter is documented in [`docs/PARAMETERS.md`](docs/PARAMETERS.md)
and settable by environment variable:

```bash
HD_CONFIDENCE=0.4 python cli.py video clip.mp4
```

---

## Engineering notes

**Duplicate suppression.** OpenCV's `dnn_DetectionModel` accepts an
`nmsThreshold` argument that has no effect with this model — verified by
sweeping it from 0.6 to 0.1 and receiving identical output at every value.
Left unhandled, the network reports the same person several times from
overlapping anchor boxes.

Suppression is therefore implemented directly, scoring candidate pairs on
both IoU **and** containment. Containment matters: a nested duplicate can
sit 95% inside a larger box while scoring only 0.28 IoU, so an IoU-only
rule keeps it.

| `samples/manbenz.png` | people | cars |
|---|---:|---:|
| raw network output | 3 | 2 |
| after suppression | **1** | 2 |

**Tracking.** Greedy IoU matching with a centroid fallback and an
occlusion tolerance, so a person walking behind an obstruction keeps their
identity rather than being counted twice on reappearance.

**Evaluation.** Detection quality is measured rather than assumed:
precision, recall and F1 with IoU matching against labelled frames, plus a
minimum-height filter so distant figures a few pixels tall are excluded
from both predictions and ground truth.

**Performance.** 34 fps end to end (detection + tracking) on CPU at
854×480. YOLOv8n runs inference ~17% faster than SSD MobileNet
(24.6 vs 29.6 ms/frame).

---

## Development log

[`docs/FINDINGS.md`](docs/FINDINGS.md) is an engineering log kept during
development — measurements taken, defects traced, and calibration work on
the evaluation set. It is a working document, not a summary of the
project's capability.

---

## Tests

```bash
python -m pytest tests/ -q     # 54 tests
```

Suppression, tracking, counting geometry, motion estimation and evaluation
metrics are all tested against synthetic inputs, so failures point at logic
rather than at model drift. No weights download needed to run them.

---

## Repository

```
humandetect/     the package
models/          network config + COCO labels (weights auto-download)
samples/         test image
scripts/         benchmarking and diagnostic tools
notebooks/       the original Comviva notebook
docs/            parameters and development log
tests/           54 tests
```

MIT licensed. Note that the optional YOLOv8 backend is AGPL-3.0 via
ultralytics; the default SSD path carries no such restriction.

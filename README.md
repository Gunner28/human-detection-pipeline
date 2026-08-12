# Human Detection Pipeline

Person detection and tracking in video, built on an SSD MobileNet v3
pipeline originally written during an internship at Comviva
(Oct 2019 – Oct 2022), and since rebuilt with a measurement harness around
it.

The interesting part of this repository is not the detector. It is
[`docs/FINDINGS.md`](docs/FINDINGS.md) — a running record of what was
measured, what broke, and which of my own conclusions turned out to be
wrong.

## Try it on your own video

```bash
git clone https://github.com/Gunner28/human-detection-pipeline
cd human-detection-pipeline
pip install -r requirements.txt

python cli.py count /path/to/your_video.mp4
```

That is the whole setup. The detection weights (13 MB) download themselves
on first run — no separate download step, no configuration, no editing
paths. Point it at any video file.

```bash
python cli.py count clip.mp4                      # footfall across the middle
python cli.py count clip.mp4 -l horizontal:0.7    # line at 70% of the height
python cli.py count clip.mp4 -l vertical:0.4      # vertical line instead
python cli.py count clip.mp4 -l 100,300,700,300   # exact pixel coordinates
python cli.py count clip.mp4 -o annotated.mp4     # see the line and the boxes
python cli.py count clip.mp4 --json               # machine-readable output

python cli.py image  photo.jpg                    # detect in a still
python cli.py video  clip.mp4 -o out.mp4          # annotate every frame
python cli.py analyse clip.mp4                    # tracking metrics
python cli.py webcam                              # live from a camera
```

**Recommended footage:** a fixed camera pointed at a doorway or corridor.
A tripod removes camera motion, a doorway makes the counting line mean
something physical, and walking through a known number of times gives you
ground truth for free.

---

## Counting people, not detections

`count` is the metric to trust. It counts tracks crossing a line rather
than trying to tally unique people, because unique-person counting proved
unmeasurable here — it ranged from 95 to 483 on identical footage purely
from tracker settings ([finding #2](docs/FINDINGS.md)).

Crossings degrade gracefully. If one person is fragmented into three
tracks, only the fragment spanning the line is counted; the other two
contribute nothing. This is how commercial footfall systems work.

---

## Origin

The brief at Comviva was audience targeting for banking and telecom ads.
The original deliverable was a notebook — preserved unedited at
[`notebooks/original_comviva_notebook.ipynb`](notebooks/original_comviva_notebook.ipynb)
— that loaded a TensorFlow frozen graph through OpenCV's DNN module and
ran detection over an image, a video, and a webcam feed.

This repository keeps that pipeline and adds the parts needed to know
whether it actually works.

---

## What it does now

**Detection.** SSD MobileNet v3 (COCO, 80 classes) through `cv2.dnn`, so
no deep-learning runtime is required. YOLOv8 is available as an
alternative backend.

**Duplicate suppression.** OpenCV's own `nmsThreshold` is a no-op with this
model — verified by sweeping it from 0.6 to 0.1 on a one-person image and
getting an identical five detections every time. Suppression is therefore
implemented here, using IoU *and* containment, because one duplicate pair
was 95% nested at only 0.28 IoU.

| `samples/manbenz.png` | people | cars |
|---|---:|---:|
| raw model output | 3 | 2 |
| after suppression | 1 | 2 |

**Tracking.** Greedy IoU matching with a centroid fallback, giving stable
IDs across frames — the difference between "5.07 people per frame" and
"N unique people, mean dwell X seconds".

**Evaluation.** Box-level precision/recall/F1 and count-level error against
labelled frames, with a minimum-height filter so distant pedestrians a few
pixels tall are excluded from ground truth and predictions alike.

---

## Honest status

**No accuracy figure from this repository is currently fit to quote.**

A first evaluation suggested the detector missed 20% of large people. It
was then found that the *labels* were wrong, not the detector — two
independent models agreed on 9 people in a frame labelled 5, and visual
review confirmed the models were right. That finding is written up in full
as [finding #5](docs/FINDINGS.md), along with the claims it withdraws.

What is measured and does hold:

| | |
|---|---|
| Duplicate suppression | 25,143 raw boxes to 20,765 over 4,094 frames |
| Frames containing a person | 75.5% |
| Throughput, detect + track | 34.2 fps, CPU, 854x480 |
| YOLOv8n vs SSD, speed | 24.6 vs 29.6 ms/frame |
| Unique-person count | **unreliable** — varies 95 to 483 with tracker settings |

The unique-person count is not yet a measurement. It is a tuning artifact,
and the parameter sweep that shows this is in the findings.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh        # fetches the 13 MB frozen graph
```

Optional second backend:

```bash
pip install ultralytics                # YOLOv8; note AGPL-3.0
```

---

## Layout

```
humandetect/
  config.py      tunables, all environment-overridable
  detector.py    SSD inference + duplicate suppression
  backends.py    swappable SSD / YOLO backends
  tracking.py    identity across frames
  analytics.py   unique people, dwell, busiest window
  evaluate.py    box-level and count-level metrics
scripts/
  download_models.sh    fetch weights
  verify_video.py       full-video detection pass
  sweep_tracking.py     tracker parameter sensitivity
  detect_cuts.py        shot-boundary detection
  make_eval_frames.py   sample frames for labelling
  compare_backends.py   SSD vs YOLO on the same ground truth
notebooks/       the original Comviva notebook, unedited
docs/FINDINGS.md the record of what was measured and what broke
tests/           29 tests; no model download required
```

---

## Tests

```bash
python -m pytest tests/ -q
```

They run without downloading weights: the metrics, suppression and
tracking logic are tested against synthetic inputs, so a failure means the
logic is wrong rather than that a model behaved differently today.

---

## Known limitations

- **Tracking has no motion model.** People crossing paths can swap IDs, and
  camera movement breaks matching — the dominant cause of the inflated
  unique-person count.
- **Ground truth is inadequate.** Three frames, one annotator, count-level.
  Being redone box-level with more frames.
- **The sample clip is edited footage** with roughly three hard cuts and
  heavy handheld motion. Tracking assumes continuity and does not hold
  across a cut.

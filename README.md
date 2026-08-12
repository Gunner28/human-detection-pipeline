# Human Detection Pipeline

Detect people in video, follow each person across frames, and report where and
when they appeared — measured against a public benchmark rather than asserted.

Built at Comviva (2019–2022) for audience analysis in advertising, and since
rebuilt around detection, tracking, and evaluation that can be checked by
someone else.

```bash
pip install -r requirements.txt
python cli.py video my_video.mp4 -o annotated.mp4
```

---

## Result

Scored on the **MOT17** train split — 7 sequences, 5,316 frames, 112,297
annotated pedestrian boxes, 546 distinct identities. Metrics computed by
[TrackEval](https://github.com/JonathonLuiten/TrackEval), the implementation
behind the public MOTChallenge leaderboard. Detections are private (this
repository's own detector), which is a separate leaderboard from the public
detections shipped with the dataset.

| detector | input | conf | HOTA | DetA | AssA | IDF1 | MOTA | MOTP |
|---|---|---|---|---|---|---|---|---|
| SSD MobileNet v3 | 320 fixed | 0.50 | 15.62 | 13.41 | 18.94 | 15.38 | 3.47 | 71.35 |
| **YOLOv8n** | **≤1280** | **0.25** | **41.99** | **41.07** | **43.45** | **50.32** | **43.87** | **79.79** |

Recall 52.8% (59,342 of 112,297 ground-truth boxes), precision 86.8%.
16.8 fps end to end on CPU, detection and tracking together.

These two rows differ in three things — detector, input size and confidence
threshold — so the table is "the pipeline as it was" against "the pipeline as it
is", not a controlled comparison. The controlled one-variable-at-a-time runs
that establish which change did what are in the
[notebook](notebooks/mot17_benchmark.ipynb) and summarised under
[Engineering notes](#engineering-notes).

Per sequence:

| sequence | HOTA | DetA | AssA | IDF1 | MOTA | ID switches |
|---|---|---|---|---|---|---|
| MOT17-02 | 30.34 | 25.66 | 36.27 | 35.35 | 27.99 | 105 |
| MOT17-04 | 47.25 | 44.99 | 50.02 | 56.77 | 48.47 | 106 |
| MOT17-05 | 42.90 | 43.64 | 42.39 | 56.74 | 48.97 | 119 |
| MOT17-09 | 44.10 | 53.30 | 36.66 | 54.28 | 60.13 | 43 |
| MOT17-10 | 36.64 | 42.45 | 32.04 | 47.92 | 46.74 | 181 |
| MOT17-11 | 46.89 | 48.50 | 45.65 | 50.35 | 43.81 | 30 |
| MOT17-13 | 32.95 | 34.52 | 32.13 | 40.26 | 36.78 | 495 |
| **combined** | **41.99** | **41.07** | **43.45** | **50.32** | **43.87** | **1079** |

Reproduce with three commands — see [Benchmark](#benchmark) below.

---

## Why these three metrics

A tracker's output is a partition of boxes into identities; so is the ground
truth. Every metric here is a different answer to how you compare two such
partitions, and the difference is not cosmetic.

**MOTA** matches per frame and counts events. False negatives and false
positives number in the thousands while identity switches number in the
hundreds, so association is structurally underweighted — MOTA is largely a
detection score wearing a tracking score's name. It can go negative, and in the
SSD baseline it does, on three sequences.

**IDF1** solves one global assignment between ground-truth and predicted
trajectories across the whole sequence. A track split in half loses roughly
half its boxes no matter how good those boxes were. This is the number that
moves when identity handling improves.

**HOTA** factorises into detection and association accuracy and takes their
geometric mean, so neither can be traded off against the other unnoticed.
Reporting `DetA` and `AssA` separately is what lets a change be attributed
rather than guessed at — and in this project that mattered twice (below).

---

## The model

**SSD MobileNet v3 Large** — a single-shot detector on a MobileNetV3
convolutional backbone, trained on COCO. Runs through OpenCV's DNN module,
which reads the TensorFlow frozen graph directly, so it deploys with neither
TensorFlow nor PyTorch installed. Useful on edge targets; also the reason its
input is fixed at 320×320, which turned out to be its central limitation.

**YOLOv8n** is the second backend, behind the same interface, and is what the
headline result uses.

| | SSD MobileNet v3 | YOLOv8n |
|---|---|---|
| Input | 320×320, fixed | 640–1280, selectable |
| Framework at runtime | OpenCV DNN only | PyTorch (ultralytics) |
| Licence | Apache 2.0 | **AGPL-3.0** |

The AGPL applies to the headline number, since it was produced with YOLOv8n.
The SSD path carries no such restriction and remains the default in `cli.py`.

---

## Pipeline

```
video ─► detection ─► duplicate suppression ─► tracking ─► analytics
         SSD / YOLO    IoU + containment        IoU + centroid   presence,
                                                 matching        dwell, flow
```

| Module | What it does |
|---|---|
| [`detector.py`](humandetect/detector.py) | CNN inference, person filtering, duplicate suppression |
| [`backends.py`](humandetect/backends.py) | SSD / YOLO behind one interface |
| [`tracking.py`](humandetect/tracking.py) | Assigns and maintains a stable identity per person |
| [`motion.py`](humandetect/motion.py) | Camera-motion compensation via sparse optical flow |
| [`mot17.py`](humandetect/mot17.py) | Reads, verifies and writes MOT17 benchmark data |
| [`benchmark.py`](humandetect/benchmark.py) | Runs the pipeline on MOT17 and scores it |
| [`analytics.py`](humandetect/analytics.py) | Dwell time, occupancy, busiest window |
| [`counting.py`](humandetect/counting.py) | Directional line-crossing counts |
| [`evaluate.py`](humandetect/evaluate.py) | Precision, recall and F1 on your own labelled frames |
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

Every parameter is documented in [`docs/PARAMETERS.md`](docs/PARAMETERS.md) and
settable by environment variable:

```bash
HD_CONFIDENCE=0.4 python cli.py video clip.mp4
```

---

## Benchmark

Download `MOT17.zip` (5.6 GB) from [motchallenge.net](https://motchallenge.net)
and unzip it to `./MOT17`. It is not in this repository and never should be.

**1. Fetch the official metric implementation.** Once.

```bash
./scripts/setup_trackeval.sh
```

TrackEval is pinned to a specific commit and patched on the way in: it still
uses `np.float`, `np.int` and `np.bool`, removed in NumPy 2.0, so without the
patch every evaluation dies on `AttributeError`. Those were aliases for the
Python builtins, so the substitution changes no behaviour — the script asserts
none remain and that the package imports.

**2. Prove the download is intact.** Before trusting any number from it.

```bash
python scripts/verify_mot17.py
```

Checks all 42 sequence folders, that each frame count matches its
`seqinfo.ini`, that frame numbering is contiguous, that every ground-truth row
parses with in-range values, and that no track id appears twice in one frame.

**3. Run and score.**

```bash
python scripts/benchmark_mot17.py --name my-run
```

Add `--backend ssd` for the baseline, `--scenes MOT17-02` for one sequence,
`--compensate` for camera-motion compensation.

[`notebooks/mot17_benchmark.ipynb`](notebooks/mot17_benchmark.ipynb) walks the
whole argument end to end, one variable at a time, with the plots.
[`notebooks/detection_walkthrough.ipynb`](notebooks/detection_walkthrough.ipynb)
covers the detection and tracking core itself, and needs no benchmark download.

### Two things about MOT17 worth knowing

**The 21 train folders are 7 videos.** Each scene is duplicated once per public
detector (DPM, FRCNN, SDP); frames and ground truth are identical across the
three. Averaging a metric over all 21 triple-counts 7 videos and yields a number
that will not match anyone else's. This repository scores the 7 and asserts the
copies agree.

**45.1% of ground-truth rows are not scored.** 92,404 of 204,701 are ignore
regions — bicycles, cars, reflections, static persons, distractors. A detection
landing on one counts as neither hit nor false positive. Skip that filter and
false positives inflate against objects the benchmark deliberately declined to
score.

---

## Engineering notes

**Duplicate suppression, and what the benchmark revealed about it.** OpenCV's
`dnn_DetectionModel` accepts an `nmsThreshold` that has no effect with this
model — verified by sweeping it from 0.6 to 0.1 and getting identical output at
every value. Suppression is therefore implemented directly, scoring candidate
pairs on IoU **and** containment, because a nested duplicate can sit 95% inside
a larger box while scoring only 0.28 IoU.

That rule was validated on a single-person test image, where a nested box really
is a duplicate. **On MOT17 it removes 10.4% of YOLO's detections, and over half
of those are real, distinct people** — someone standing behind someone else is
genuinely 70%+ contained by them, and the rule cannot tell that from a duplicate.
Disabling it recovers 505 true positives on MOT17-02 and raises DetA from 25.66
to 27.72.

It is still enabled by default, because disabling it *lowers* HOTA: the
recovered detections are heavily occluded people who appear and vanish, so the
tracker fragments on them and AssA falls from 36.27 to 29.91 — further than
DetA gained. MOTA alone would have called this a win. The honest fix is a rule
that distinguishes a duplicate from an occluded neighbour, and it is not
written yet.

**Resolution decides which people exist.** A pedestrian 40 px tall in 1080p is
13 px at an inference size of 640 and 26 px at 1280. Below roughly 16 px the
detector stops finding them, so input size is not a tuning knob — it determines
what the model can see. Inference size is capped at each sequence's native
resolution, because upscaling past native invents no detail but plenty of false
positives: MOT17-05 (640×480) run at 1280 scored 1,546 false positives against
673 at native size, costing 22.6 points of MOTA and 6.9 of HOTA — same model,
same confidence, input size the only difference.

**Camera motion is the dominant cause of inflated counts on handheld footage.**
A stationary person filmed by a moving camera has a box that jumps between
frames; overlap collapses, the track dies, and a new identity is minted from
nothing. [`motion.py`](humandetect/motion.py) estimates the global transform by
sparse optical flow with people masked out of the feature set — including them
biases the estimate toward the crowd's motion rather than the camera's.

**Tracking.** Greedy IoU matching with a centroid-distance fallback and an
occlusion tolerance, in the spirit of SORT but without a Kalman filter. The
trade-off is deliberate and visible in the numbers: no motion model means two
people crossing paths can swap identities.

---

## Known limitations

Stated because a benchmark makes them measurable, not because they are
comfortable.

- **Recall is 52.8%.** Roughly half of all annotated pedestrians are never
  detected, concentrated in small distant figures. A larger YOLO variant or
  tiled inference targets this directly.
- **MOT17-13 accounts for 495 of 1,079 identity switches** — 46% of the total,
  from 10% of the data. It has the fastest camera motion in the set, and every
  benchmark run above left camera-motion compensation *off*.
- **No appearance model.** Matching is geometric only, so identity cannot
  survive a long occlusion. This is what a pretrained Re-ID network would fix,
  and IDF1 is the metric that would show whether it did.
- **Scored on the train split.** The test split has no public ground truth and
  is server-scored only. Train-split numbers are the right basis for the
  before-and-after comparisons here, and are not interchangeable with
  leaderboard entries.

---

## Development log

[`docs/FINDINGS.md`](docs/FINDINGS.md) records measurements taken and defects
traced during development, including the ones that contradicted an earlier
assumption. It is a working log, not a summary of capability.

---

## Tests

```bash
python -m pytest tests/ -q     # 54 tests
```

Suppression, tracking, counting geometry, motion estimation and evaluation
metrics are tested against synthetic inputs, so a failure points at logic rather
than at model drift. No weights download needed to run them.

---

## Repository

```
humandetect/     the package
models/          network config + COCO labels (weights auto-download)
samples/         test image and clip
scripts/         benchmark, setup and diagnostic tools
notebooks/       detection walkthrough and benchmark walkthrough
docs/            parameters and development log
tests/           54 tests
```

MIT licensed. The optional YOLO backend is AGPL-3.0 via ultralytics; the
default SSD path carries no such restriction.

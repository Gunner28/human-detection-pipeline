# Parameters

Every tunable lives in [`humandetect/config.py`](../humandetect/config.py)
and reads an environment variable, so nothing needs editing to change
behaviour:

```bash
HD_CONFIDENCE=0.4 python cli.py count my_video.mp4
```

---

## Detection

| Parameter | Env var | Default | What it does |
|---|---|---|---|
| `CONFIDENCE_THRESHOLD` | `HD_CONFIDENCE` | `0.5` | Minimum score for a box to be reported. Lower finds more people and more false positives. |
| `IOU_THRESHOLD` | `HD_IOU` | `0.45` | Two boxes overlapping by at least this much are treated as the same object. |
| `CONTAINMENT_THRESHOLD` | `HD_CONTAINMENT` | `0.7` | A box this fraction *inside* another is a duplicate, even at low overlap. |

**Why two thresholds.** Overlap alone misses nested duplicates. On the
sample image, one pair of boxes on the same man was 95% nested but only
0.28 overlap — overlap-only suppression kept both. See
[finding #1](FINDINGS.md).

**Tuning confidence.** Start at 0.5. Raise it if you get boxes on things
that are not people; lower it if people are being missed. Compare with:

```bash
python cli.py --raw --all-classes image photo.jpg   # unfiltered model output
```

---

## Model input

Fixed by how the network was trained. Changing these breaks detection.

| Parameter | Value | Why |
|---|---|---|
| `INPUT_SIZE` | `(320, 320)` | Resolution SSD MobileNet v3 expects. |
| `INPUT_SCALE` | `1/127.5` | Scales pixels to the range the network was trained on. |
| `INPUT_MEAN` | `(127.5, 127.5, 127.5)` | Centres pixel values on zero, giving `[-1, 1]`. |
| `PERSON_CLASS_ID` | `1` | COCO class index for `person`, as this model emits it (1-indexed). |

---

## Tracking

| Parameter | Env var | Default | What it does |
|---|---|---|---|
| `TRACK_IOU_THRESHOLD` | `HD_TRACK_IOU` | `0.3` | Overlap needed for a new box to continue an existing track. |
| `TRACK_DISTANCE_FRACTION` | `HD_TRACK_DISTANCE` | `0.08` | Fallback when boxes do not overlap: match if centres are within this fraction of the frame diagonal. |
| `TRACK_MAX_MISSING` | `HD_TRACK_MAX_MISSING` | `15` | Frames a person may vanish before the track closes. At 30fps this is half a second — enough to survive walking behind a sign. |
| `TRACK_MIN_FRAMES` | `HD_TRACK_MIN_FRAMES` | `5` | Tracks shorter than this are discarded as detector flicker. |

**These two matter more than they look.** On the sample clip, the
unique-person count ranged from **95 to 483** depending only on
`TRACK_MAX_MISSING` and `TRACK_MIN_FRAMES` — which is why unique-person
counting is not used as a headline metric. See [finding #2](FINDINGS.md).

Raising `TRACK_MAX_MISSING` merges fragments back together but also keeps
dead tracks alive as ghosts, which inflates concurrent-occupancy figures.
There is no setting that is simply correct; it depends on the footage.

`python cli.py count` avoids the problem: crossings are far less sensitive
to fragmentation than unique-person counts.

---

## Counting

Set on the command line rather than in config, since the right line depends
entirely on the scene.

| Option | Example | Meaning |
|---|---|---|
| `--line horizontal` | default | Straight across the middle. |
| `--line horizontal:0.7` | | Across at 70% of the frame height. |
| `--line vertical:0.4` | | Down at 40% of the frame width. |
| `--line 100,300,700,300` | | Explicit pixel coordinates `x1,y1,x2,y2`. |
| `--positive-is out` | | Flip which direction is labelled "in". |

Put the line where people must physically pass — a doorway, a gate, the
narrow point of a corridor. A line across open space produces a number
without meaning.

---

## Camera-motion compensation

On by default; disable with `--no-motion-compensation`.

Estimates how the whole frame moved between frames and shifts tracks by
that amount before matching, so a moving camera does not look like moving
people.

**Honest note:** on the bundled sample clip this makes no measurable
difference — 568 crossings with it, 568 without, and measured motion of
only 1.22 px/frame. It is verified correct against synthetic data and
should help genuinely shaky footage, but it did not help here. See
[finding #7](FINDINGS.md).

---

## Backends

| Value | Model | Notes |
|---|---|---|
| `ssd` | SSD MobileNet v3 | Default. No deep-learning runtime needed. |
| `yolo` | YOLOv8n | Requires `pip install ultralytics`. **AGPL-3.0.** |
| `yolo:yolov8s.pt` | Larger YOLO | Any ultralytics weight name. |

Benchmarked against each other in
[`outputs/backend_comparison.json`](../outputs/backend_comparison.json).
The comparison was **inconclusive** — the differences were smaller than the
error in the ground-truth labels ([finding #6](FINDINGS.md)). YOLOv8n was
consistently faster (24.6 vs 29.6 ms/frame).

---

## Evaluation

| Parameter | Default | What it does |
|---|---|---|
| `DEFAULT_MIN_HEIGHT` | `60` px | People shorter than this are excluded from ground truth *and* predictions. |

Distant pedestrians a few pixels tall cannot be counted reliably by eye, so
scoring the detector against them measures the labeller, not the model.
COCO handles this with size categories; crowd datasets use ignore regions.

Every accuracy figure therefore carries the qualifier *"for people at least
60px tall"*, and that qualifier is not optional when quoting it.

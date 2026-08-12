# Development log

Measurements taken while building and calibrating this pipeline: what was
tested, what the numbers were, and how the evaluation set was refined.

Kept because the reasoning is reusable — the calibration issues here are
the ordinary ones in any detection project (duplicate boxes, tracker
parameter sensitivity, annotation consistency, footage suitability).

---

## 1. OpenCV's NMS parameter is a no-op with this model

**Date:** 2026-08-11 · **Status:** fixed

`cv2.dnn_DetectionModel.detect(nmsThreshold=...)` does not suppress
anything with SSD MobileNet v3. Verified by sweeping the threshold from
0.6 down to 0.1 on `samples/manbenz.png` (one man, one car) and getting an
identical five detections at every value.

The duplicate boxes were real: two pairs at IoU 0.508 and 0.553, well past
any sane threshold.

| | people | cars |
|---|---|---|
| Raw model output | 3 | 2 |
| After suppression | 1 | 2 |

Fixed in `detector.suppress()`, which applies greedy per-class NMS using
**both** IoU and containment. Containment matters: one duplicate pair was
95% nested but only 0.28 IoU, so an IoU-only rule would have kept it.

Across the full 4,094-frame video the effect is smaller than on that
single image — 25,143 raw boxes to 20,765 deduplicated, a factor of 1.21.
The 3x seen on `manbenz.png` is not representative of the general case.

---

## 2. Tracker parameter sensitivity

**Date:** 2026-08-11 · **Status:** open — blocks any headline claim

Tracking was added to convert per-frame detections into people. It
produces metrics of the right *kind* (unique people, dwell, busiest
window) but not yet of usable *accuracy*.

Sweeping the two tracker parameters over cached detections from the same
4,094 frames:

| `max_missing` | `min_frames` | unique people | mean dwell | median dwell | peak |
|---:|---:|---:|---:|---:|---:|
| 5 | 5 | **483** | 1.41s | 0.70s | 18 |
| 5 | 15 | 295 | 2.12s | 1.43s | 17 |
| 5 | 30 | 194 | 2.87s | 2.10s | 16 |
| 15 | 5 | 342 | 2.01s | 1.00s | 18 |
| 15 | 15 | 241 | 2.73s | 1.73s | 18 |
| 15 | 30 | 173 | 3.53s | 2.47s | 18 |
| 30 | 5 | 234 | 2.95s | 1.52s | 21 |
| 30 | 15 | 184 | 3.66s | 2.22s | 21 |
| 30 | 30 | 142 | 4.53s | 3.03s | 20 |
| 60 | 5 | 171 | 4.04s | 1.70s | 25 |
| 60 | 15 | 135 | 5.03s | 2.83s | 25 |
| 60 | 30 | 108 | 6.11s | 3.67s | 24 |
| 90 | 5 | 145 | 4.76s | 2.20s | 25 |
| 90 | 15 | 112 | 6.07s | 3.52s | 25 |
| 90 | 30 | **95** | 7.03s | 4.73s | 24 |

**The count spans 95 to 483 — a 5x range — from parameter choice alone.**
The default configuration lands on 342. Nothing justifies that number over
any other row.

Two things the table shows:

1. **Fragmentation is heavy.** Unique count falls monotonically as
   `max_missing` grows, and has still not plateaued at 90 frames (three
   seconds of tolerance). If tracking were clean, the count would stabilise
   once the tolerance exceeded typical occlusion length. It doesn't, so
   single people are being split into multiple tracks.

2. **Raising tolerance trades one error for another.** Peak concurrent
   *rises* from 18 to 25 as `max_missing` grows, because tracks that should
   have closed linger as ghosts. There is no setting that is simply
   "correct".

**Root cause:** SSD MobileNet v3 at 320x320 flickers on small and
partially occluded figures in 854x480 footage. The tracker has no motion
model, so a few missed frames end a track.

**What this blocks:** unique-person, footfall and dwell figures from this
pipeline should be quoted until it is validated. The busiest-window
position is likely more robust than the counts, since it depends on
relative occupancy rather than absolute identity — but that is an
expectation, not yet a measurement.

**Next:** label a sample of frames for ground truth, measure detection
precision/recall, and tune against evidence rather than intuition. A
motion model (Kalman) and a stronger detector are candidate fixes, but
should be chosen after measurement, not before.

---

## 3. Shot-boundary analysis of the test footage

**Date:** 2026-08-11 · **Status:** diagnosed

Sampling twelve frames for labelling revealed the clip is not one
continuous take. Three sampled frames showed a blurred close-up under an
arcade (10.5s), a ground-level shot of legs (31.4s), and a wide high
street (62.8s) — three different scenes.

That mattered because tracking assumes continuity. Across a hard cut every
person is unmatched, so each cut manufactures a fresh set of "unique
people". The initial hypothesis was that cuts were inflating the count more
than detector flicker.

**That hypothesis was wrong.** Sweeping the cut-detection threshold
(mean absolute frame difference on a 160x90 grayscale downscale, expressed
as a multiple of the median difference):

| threshold multiplier | raw spikes | boundaries | median shot |
|---:|---:|---:|---:|
| 4x | 680 | 152 | 0.2s |
| 6x | 362 | 85 | 0.2s |
| 8x | 241 | 53 | 0.2s |
| 12x | 94 | 25 | 0.2s |
| 20x | 9 | **3** | **26.2s** |

The count collapses from 152 to 3 as the threshold rises, and the median
shot length only becomes plausible at the strictest setting. A montage
with a genuine 0.2s median shot length would be unwatchable. So the low
thresholds were not finding cuts — they were finding **camera motion**.

Distribution supports this: median frame difference 1.46, 95th percentile
13.08, max 74.81. The long tail is handheld movement, not editing.

**Revised diagnosis of the inflated person count**, in order of
contribution:

1. **Camera motion (major).** The tracker matches boxes between frames by
   overlap. When the camera itself moves, a stationary person's box jumps
   between frames and the overlap collapses, so the track breaks and a new
   identity is minted. This is the dominant failure.
2. **Detector flicker (major).** SSD MobileNet v3 at 320x320 loses small
   and partially occluded figures for a frame or two at a time.
3. **Scene cuts (minor).** Roughly three genuine cuts across the clip.
   Three resets cannot account for a count of 342.

**Consequences for the fix.** Camera-motion compensation — estimating
global frame-to-frame motion and warping track positions before matching —
addresses the largest cause and is cheap. A motion model (Kalman) helps
with the second. Neither is worth tuning blind, which is why measurement
still comes first.

**Consequence for evaluation.** Ground truth is harder here than expected.
The wide-street frame contains 40+ pedestrians, many only a few pixels
tall; the ground-level frame shows legs with no heads at all. Counting
those reliably is not possible even by eye. The evaluation protocol
therefore needs an explicit minimum-size criterion, with anything smaller
excluded from both ground truth and predictions — the same approach COCO
takes with its size categories and that crowd datasets take with ignore
regions.

---

## 4. First accuracy pass (superseded by #5)

**Date:** 2026-08-11 · **Status:** measured (pilot, N=3 frames)

First real accuracy number for the pipeline. Ground truth is **count-level**
on people at least **60 pixels tall** in the source resolution, labelled by
a single annotator with a 60px reference bar rendered into each frame as a
yardstick. Borderline cases carry roughly +/-1 person of uncertainty.

Three frames, 15 labelled people in total:

| confidence | detected | actual | ratio | MAE / frame | bias | per-frame detected |
|---:|---:|---:|---:|---:|---:|---|
| 0.4 | 18 | 15 | 1.20x | **1.67** | +1.00 | 9, 4, 5 |
| **0.5** (default) | 12 | 15 | **0.80x** | **3.00** | -1.00 | 8, 1, 3 |
| 0.6 | 7 | 15 | 0.47x | 2.67 | -2.67 | 4, 1, 2 |
| 0.7 | 2 | 15 | 0.13x | 4.33 | -4.33 | 1, 1, 0 |

Ground truth per frame was 5, 5, 5.

**At the default confidence of 0.5 the detector finds 80% of large people
and has a mean absolute error of 3.0 people per frame — on a true count of
5. That is a 60% error.**

Worse than the average is the inconsistency. On the same setting the
detector reported **8 where there were 5** (over by 3) on the wide high
street, then **1 where there were 5** (under by 4) on the New York street
scene. That second frame has heavy motion blur and foreground people
cropped by the frame edge; the model collapses on it.

**Confidence 0.4 beats the 0.5 default** on this sample: MAE 1.67 versus
3.00. Not adopted as the new default yet — three frames is a pilot, not a
validation set.

**This explains finding #2.** Tracking fragmentation was diagnosed as
"detector flicker" without a number attached. Now there is one: the
detector misses roughly one person in five, frame to frame, among people
large enough to count easily. A tracker matching boxes between consecutive
frames cannot survive that miss rate — every miss is a chance to break a
track and mint a new identity. The inflated unique-person count is a
downstream symptom of per-frame detection quality, not primarily a tracker
bug.

**Limits of this result, stated plainly:**
- Three frames, one annotator, no inter-annotator agreement check.
- Count-level, not box-level: a detector could get the right count with
  boxes in the wrong places and score perfectly here.
- Only people >= 60px tall. Says nothing about the distant pedestrians
  that make up most of the wide-street frames.

**What it unlocks.** There is now a measuring stick. Swapping SSD MobileNet
for a modern detector can be judged on evidence instead of reputation, and
the confidence threshold can be tuned against data. That was the whole
argument for building this before changing models.

---

## 5. Calibrating the evaluation set

**Date:** 2026-08-11 · **Status:** corrects finding #4

Benchmarking a second detector exposed an error in the labels, not in the
models.

On frame 1884 both SSD MobileNet and YOLOv8n independently reported **9**
people at 60px or larger. The hand label said **5**. Two unrelated
architectures agreeing against a single human label is a strong signal
that the label is the problem, so the frame was re-examined with the
detections drawn on.

**The detections were right.** The boxes the label had missed sat on real
mid-ground pedestrians 65-80px tall — genuinely above the 60px threshold.
They had been dismissed during labelling as "borderline, probably too
small". The true count is roughly 8-9, not 5.

**Consequences:**

* Finding #4's headline — *"the detector finds 80% of large people"* — is
  **not reliable**. The apparent 20% miss rate was substantially an
  artifact of labels biased low. On this frame SSD at confidence 0.5
  reported 8 against a true count of ~8-9, which is close to correct, not
  a 3-person over-count.
* The claim that confidence 0.4 beats the 0.5 default also falls, since it
  rested on the same labels.
* The tracking-fragmentation explanation in finding #4 — that a measured
  per-frame miss rate explains broken tracks — loses its measured
  underpinning. Fragmentation is still real and still visible in the
  parameter sweep, but the specific miss-rate figure offered as its cause
  is withdrawn.

**Root cause of the labelling error:** a single annotator applying a size
threshold by eye, with no second opinion and no calibration pass. Borderline
cases were resolved conservatively and consistently in one direction, which
turns random error into systematic bias.

**Fixes for the next attempt:**
1. Label with detections hidden, then *review* with them shown, and record
   disagreements rather than silently deferring to either side.
2. Enforce the size threshold by measurement, not by eye — draw candidate
   boxes and check heights numerically.
3. Label box-level, not count-level, so a right-for-the-wrong-reasons count
   cannot pass.
4. More frames. Three cannot separate models whose true difference is
   smaller than the labelling error.

**The wider lesson, which is the point of writing this down:** the harness
worked exactly as intended. It was built to catch unsupported claims, and
the first unsupported claim it caught was one of mine. An evaluation set is
itself a measurement, with its own error bars, and it must be validated
before anything is concluded from it.

---

## 6. SSD MobileNet v3 vs YOLOv8n benchmark

**Date:** 2026-08-11 · **Status:** inconclusive, by design worth recording

Same three frames, same size filter, same suppression, only the model
changed. Numbers below are scored against the labels now known to be
unreliable (finding #5), so treat the accuracy columns as indicative only.
The timing column is unaffected.

| model | conf | detected | ratio | MAE | ms/frame |
|---|---:|---:|---:|---:|---:|
| SSD MobileNet v3 | 0.4 | 18 | 1.20x | 1.67 | 30.1 |
| SSD MobileNet v3 | 0.5 | 12 | 0.80x | 3.00 | 29.6 |
| YOLOv8n | 0.4 | 18 | 1.20x | 2.33 | 24.9 |
| YOLOv8n | 0.5 | 14 | 0.93x | 3.00 | 24.6 |

**No clear winner on accuracy.** The differences are smaller than the
labelling error, so the comparison cannot discriminate. Reporting "YOLO is
better" from this data would be exactly the kind of unsupported claim the
harness exists to prevent.

Two observations that do hold:

* **YOLOv8n is faster here** — about 25ms versus 30ms per frame on CPU,
  consistently across every confidence setting.
* **Both models fail identically on frame 2512**, each finding 1 person
  where there are about 5. That frame has heavy motion blur and figures
  cropped at the edge. A shared failure across architectures points at
  intrinsic frame difficulty rather than a model weakness, and no amount of
  model-swapping will fix it.

**Licensing, which matters for a portfolio:** ultralytics YOLOv8 is
AGPL-3.0. Fine for public code, but it would need replacing or a
commercial licence before shipping inside closed-source software. The SSD
path through OpenCV carries no such restriction. That is a genuine reason
to keep both backends rather than simply migrating.

---

## 7. Camera-motion compensation: measured impact on this footage

**Date:** 2026-08-11 · **Status:** corrects finding #3

Finding #3 named camera motion the largest contributor to inflated person
counts. Camera-motion compensation was then implemented specifically to
address it, and measured with an A/B run on the same clip:

| | crossings | in | out |
|---|---:|---:|---:|
| compensation off | 568 | 282 | 286 |
| compensation on | 568 | 281 | 287 |

**Effectively no change** — a difference of one crossing in each direction,
well inside noise.

The estimator also reports the motion it measured: **1.22 px/frame on
average**. That is small. The hypothesis assumed large frame-to-frame
displacement breaking box overlap; the measurement says the displacement is
about a pixel.

So finding #3's ranking was wrong in the same way finding #4's accuracy
number was wrong: it was reasoning from a plausible mechanism rather than
from a measurement of that mechanism. The frame-difference statistic that
motivated it (95th percentile 13.08) measures *pixel intensity change*,
which rises with subject movement, lighting and blur — not only camera
translation. Treating it as a proxy for camera motion was the error.

**What still stands:** the unique-person count is unreliable (finding #2 —
that one is a direct measurement, a 95-to-483 range from a parameter
sweep). What is no longer supported is any confident claim about *why*.

**Current honest position on the cause:** unknown, with detector flicker
the leading remaining candidate and the clip's edited, wildly varying shot
structure a likely contributor. Establishing this properly needs per-track
diagnostics — where tracks die, and what the detector was doing at that
moment — not another mechanism-first hypothesis.

**The compensation code is kept**, for two reasons. It is verified correct
against synthetic data (it recovers known translations to within 3px, and
degrades to identity rather than inventing motion when it cannot estimate),
and it is off-the-shelf useful for genuinely shaky footage. It simply does
not help *this* clip, and the repository should say so.

**Method note worth carrying forward:** three hypotheses have now been
tested here and three have been wrong — scene cuts, labelling accuracy,
camera motion. The pattern is consistent: each was a mechanism that sounded
right and was adopted before it was measured. The harness catches these,
which is the argument for building it first, but the cheaper lesson is to
measure the mechanism directly before designing a fix for it.

---

## 8. Line-crossing counting

**Date:** 2026-08-11 · **Status:** implemented

Unique-person counting is not salvageable as a headline metric on this
footage (finding #2). Rather than keep tuning it, the metric itself was
changed.

A virtual tripwire counts tracks crossing a line. Its advantage is
structural: if one person is fragmented into three tracks, only the
fragment that actually spans the line contributes, so the count degrades
gracefully where unique-person counting collapses. This is how commercial
footfall systems work, and it maps directly onto the question a retail or
out-of-home advertising client asks — how many people came through, and
which way were they going.

The geometry is stricter than a side test: a crossing requires both a
change of side *and* a genuine segment intersection, so a track passing
beyond the end of the line is not counted.

**The 568 crossings measured on the sample clip are not a footfall
figure.** A horizontal line across the middle of an edited montage of
unrelated street scenes has no physical meaning. The number demonstrates
the mechanism runs; it measures nothing about any real place.

Getting a defensible number requires footage the metric suits: a fixed
camera on a doorway or corridor, where the line corresponds to something
real and the true count is known by observation.

---

## 9. Throughput

**Date:** 2026-08-11 · **Status:** measured

| Pass | Throughput |
|---|---|
| Detection only, writing annotated video | 16.8 fps |
| Detection + tracking, no video write | 34.2 fps |

CPU only, Apple Silicon, 854x480 input. The difference is mostly the video
encode and the second detection pass in the Phase 1 verification script,
not the tracker, which is cheap.

---

## 10. MOT17: replacing hand-labelled ground truth

**Date:** 2026-08-12 · **Status:** measured

Findings #4 and #5 both hit the same ceiling: a few hundred frames labelled
by one annotator, with no way to compare the result against anyone else's.
Finding #2 recorded a unique-person count ranging from 95 to 483 across
tracker parameters, with no basis for choosing between them.

MOT17 removes that ceiling. Downloaded 5.6 GB from motchallenge.net and
verified with `scripts/verify_mot17.py`:

| | |
|---|---|
| Train scenes | 7 (21 folders = 7 scenes x 3 public detectors) |
| Frames | 5,316 |
| Ground-truth rows | 204,701 |
| Evaluated pedestrian boxes | 112,297 |
| Distinct identities | 546 |

Two properties of the dataset are easy to get wrong and both change the
number reported:

**The 21 train folders are 7 videos.** Frames and ground truth are
identical across the three detector copies. Averaging over 21 triple-counts
7 videos. `mot17.verify()` asserts the copies agree, which is what
justifies scoring only the 7.

**45.1% of ground-truth rows are not scored.** 92,404 of 204,701 are
ignore regions — occluders, bicycles, cars, static persons, distractors,
reflections. A detection landing on one counts as neither hit nor false
positive. Skipping that filter inflates false positives against objects the
benchmark deliberately declined to score.

Metrics come from TrackEval, pinned at commit `12c8791`. It is unmaintained
against NumPy 2: 81 uses of `np.float`, `np.int` and `np.bool`, all removed
in 2.0, so every evaluation died on `AttributeError` until patched.
`scripts/setup_trackeval.sh` applies the substitution and asserts none
remain.

---

## 11. The baseline was a precision problem, not a recall problem

**Date:** 2026-08-12 · **Status:** measured, hypothesis disproved

Predicted that input resolution was the dominant limitation and would show
up as recall. It was not, and it did not.

MOT17-02, tracker and all other settings identical:

| run | HOTA | DetA | AssA | IDF1 | MOTA | TP | FN | FP | IDSW |
|---|---|---|---|---|---|---|---|---|---|
| SSD, 320 | 11.21 | 11.06 | 11.54 | 11.60 | −2.89 | 2743 | 15838 | 3177 | 103 |
| YOLOv8n, 640 | 26.67 | 14.32 | 49.72 | 24.60 | 16.37 | 3143 | 15438 | **78** | 24 |
| YOLOv8n, 1280 | 28.57 | 19.33 | 42.42 | 29.00 | 22.19 | 4293 | 14288 | 122 | 47 |

From SSD-320 to YOLO-640, recall moves 14.8% to 16.9% — barely. False
positives fall from 3,177 to 78, a fortyfold reduction, and that is what
takes MOTA from negative to positive. The baseline's problem was that over
half of what it emitted was not a person. MOTP rising 69.5 to 83.8 says the
same thing from the other side: the boxes it did produce were poorly placed,
and a box near the IoU 0.5 boundary is charged as both a false positive and
a false negative.

Resolution is real but secondary: 640 to 1280 is where recall improves
(16.9% to 23.1%), at roughly half the throughput.

**AssA falls as the detector improves** — 49.72 at 640, 42.42 at 1280. Not
noise: more detections means more simultaneous candidates and more chances
to confuse two of them, so ID switches rise 24 to 47. HOTA still climbs
because DetA gains more than AssA loses. A single MOTA figure would have
hidden the trade entirely.

---

## 12. Containment suppression deletes real people in crowds

**Date:** 2026-08-12 · **Status:** measured, defect confirmed, not fixed

Finding #1 established that suppression must score containment as well as
IoU, because a nested duplicate can sit 95% inside a larger box at only
0.28 IoU. That was validated on `samples/manbenz.png` — one person, one car.

On MOT17 the same rule is wrong. A person standing behind another is
*genuinely* 70%+ contained by them, and the rule cannot distinguish that
from a duplicate box.

MOT17-02, YOLOv8n at 1280, confidence 0.25:

| | detections | HOTA | DetA | AssA | TP | FP |
|---|---|---|---|---|---|---|
| containment on | 7,578 | **30.34** | 25.66 | 36.27 | 6040 | 734 |
| containment off | 8,454 | 28.54 | **27.72** | 29.91 | **6545** | 1023 |

The rule removes 876 of 8,454 detections (10.4%), and disabling it recovers
505 true positives. This contradicts the previous README claim that on YOLO
output the rule "almost never removes anything".

**It is still enabled by default, because disabling it lowers HOTA.** The
recovered detections are heavily occluded people who appear and vanish, so
the tracker fragments on them: AssA falls further than DetA gains. MOTA
alone would have called this a win and shipped a regression.

The honest fix is a rule that distinguishes a duplicate from an occluded
neighbour — comparing aspect ratio and size, or skipping containment
entirely on backends that already apply working NMS. Not written yet.

Note on method: the first attempt to measure this toggled a module constant
and called `importlib.reload()`. That silently did nothing, because
`backends.py` binds `suppress` at import time and keeps the old reference —
both runs produced byte-identical scores, which reads as evidence of no
effect. `containment_threshold` is now a parameter threaded through
`BenchmarkConfig`, and the difference is real.

---

## 13. Inference size must be capped at native resolution

**Date:** 2026-08-12 · **Status:** measured, fixed

Running every sequence at `imgsz=1280` upscales the one 640x480 sequence
past its native size. This invents no detail and plenty of false positives.

MOT17-05, YOLOv8n, confidence 0.25, input size the only variable:

| imgsz | HOTA | MOTA | FP |
|---|---|---|---|
| 1280 (upscaled) | 36.04 | 26.40 | 1546 |
| 640 (native) | **42.90** | **48.97** | **673** |

22.6 points of MOTA and 6.9 of HOTA, from one wrong default.

`BenchmarkConfig.effective_imgsz()` now caps at each sequence's native
resolution, rounded to the multiple of 32 YOLO expects. Found only because
the per-sequence table showed MOT17-05 scoring *worse* than the SSD
baseline it was supposed to beat — a combined figure would have buried it.

**Result after the fix**, all 7 train scenes, YOLOv8n, confidence 0.25,
imgsz capped at native:

| | HOTA | DetA | AssA | IDF1 | MOTA | MOTP |
|---|---|---|---|---|---|---|
| SSD baseline | 15.62 | 13.41 | 18.94 | 15.38 | 3.47 | 71.35 |
| YOLOv8n | **41.99** | 41.07 | 43.45 | **50.32** | **43.87** | 79.79 |

Recall 52.8% (59,342 of 112,297), precision 86.8%, 16.8 fps on CPU.

Largest remaining losses, both addressable with code already in the
repository: recall at 52.8%, and MOT17-13 contributing 495 of 1,079 ID
switches — 46% of the total from 10% of the data, on the fastest camera
motion in the set, with `compensate=False` throughout.

---

## 14. Baseline notebook: what was changed, and what was not

**Date:** 2026-08-12 · **Status:** verified, then made unverifiable by design

`notebooks/baseline_pipeline.ipynb` presents the original detection
notebook with a markdown explanation before each step. The code cells are
the original ones. That claim was mechanically verified at the time of
writing, against the original notebook as it stood before publication:

    18 code cells
    13 byte-identical to the original
     5 edited (cells 3, 6, 11, 20, 22): 4 path rewrites, 10 defect fixes

**Path rewrites (4).** The original hardcoded absolute paths on the machine
it was written on, so it could not load the model anywhere else:

    error: FAILED: fs.is_open(). Can't open ".../frozen_inference_graph.pb"

Rewritten to resolve inside this repository. No logic touched.

**Defect fixes (10).** Both capture loops were broken.

*Video loop:* the "camera not opened" fallback assigned to `ccap`, so it
opened a capture nothing read from and left `cap` closed; `raise IDError`
names something that does not exist in Python, so the guard raised
`NameError` rather than its message; and no `if not ret: break`, so at the
end of a file `read()` returned `(False, None)` and `detect(None)` raised.
The loop ran and drew boxes correctly, then always terminated by crashing —
a traceback after the work is done reads as "finished".

*Webcam loop:* the above, plus `cv2.videoCapture` (lowercase v),
`confzthreshold=` (not a parameter `detect()` accepts), `cv2.waitkey`
(lowercase k), and `confidece` (misspelled consistently, so it worked).

Six defects in one cell is not carelessness. Python resolves attribute
names when a line executes, not when it is defined, so `cv2.videoCapture`
is a valid expression until evaluated. Nobody had a second camera, the loop
body never ran, and the errors stayed invisible.

After the fix, the video loop completes all 4,094 frames and exits cleanly;
the webcam loop raises `IOError` when no camera is present, which is what
it always intended to do.

**Why this entry exists.** The check lived in a build script that read the
original notebook out of git history. That notebook was removed from every
commit before publication: it hardcoded absolute paths containing a
personal home directory, and a public repository is a poor place for one.
The check therefore cannot be re-run, and the script was deleted rather
than left to fail against a file that is no longer there.

This entry is the record. The pre-publication history, including the
original notebook, is retained privately outside the repository.

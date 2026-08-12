# Outputs

Everything the tools write lands here. Videos, cached detections and
working frames are gitignored; the small files below are committed because
they are evidence for claims made in [docs/FINDINGS.md](../docs/FINDINGS.md).

| File | What it shows |
|---|---|
| `manbenz_detected.png` | One man, one car, correctly detected after suppression |
| `busiest_frame.png` | 16 people found in a crowded street scene. **Faces blurred** — see below |
| `ground_truth.json` | Hand labels. **Stamped unreliable** — see finding #5 |
| `sweep.json` | Tracker parameter sweep: unique count ranges 95 to 483 |
| `backend_comparison.json` | SSD vs YOLOv8n on identical frames |

Generated files (gitignored): `*.mp4`, `*.pkl`, `eval_frames/`,
`label_frames/`, `mot17/`, `mot17_eval/`.

## Privacy

`busiest_frame.png` is a still from street footage: people who did not
consent to appearing in a public repository. The head region of every
detected person is blurred before publication, deliberately covering more
than a face detector would — a missed person is a privacy failure, an extra
blurred patch of pavement is not. The detection boxes stay readable, which
is what the figure exists to show.

The source clip is not committed at all, for the same reason. See
[`samples/README.md`](../samples/README.md).

## Benchmark results

`mot17/<run>/data/*.txt` and `mot17_eval/<run>/` are written by
[`scripts/benchmark_mot17.py`](../scripts/benchmark_mot17.py) and are not
committed: they are regenerable from the dataset in minutes, and the
headline numbers they produce live in [the README](../README.md) and
[docs/FINDINGS.md](../docs/FINDINGS.md) #10–13 instead.

The hand-labelled evidence above is kept because it is *not* regenerable —
it took manual annotation, and finding #5 is the record of why it was
stamped unreliable. MOT17 supersedes it for measuring detection and
tracking quality, but it still answers a question MOT17 cannot: whether the
headcount on this project's own footage is right.

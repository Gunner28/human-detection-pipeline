# Outputs

Everything the tools write lands here. Videos, cached detections and
working frames are gitignored; the small files below are committed because
they are evidence for claims made in [docs/FINDINGS.md](../docs/FINDINGS.md).

| File | What it shows |
|---|---|
| `manbenz_detected.png` | One man, one car, correctly detected after suppression |
| `check_1884.png` | The frame where two models disagreed with my labels — and were right |
| `busiest_frame.png` | 16 people found in a crowded street scene |
| `ground_truth.json` | Hand labels. **Stamped unreliable** — see finding #5 |
| `sweep.json` | Tracker parameter sweep: unique count ranges 95 to 483 |
| `backend_comparison.json` | SSD vs YOLOv8n on identical frames |

Generated files (gitignored): `*.mp4`, `*.pkl`, `eval_frames/`,
`label_frames/`.

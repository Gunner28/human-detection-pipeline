# Models

| File | Committed | What it is |
|---|---|---|
| `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` | yes, 180 KB | Network architecture, read by OpenCV |
| `coco_labels.txt` | yes, 4 KB | The 80 COCO class names, in order. `person` is line 1. |
| `frozen_inference_graph.pb` | **no**, 13 MB | The trained weights |

The weights are not committed. They download automatically the first time
you run anything, or explicitly:

```bash
bash scripts/download_models.sh
```

Source: `download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz`

## Why this model

It is the model the original Comviva work used, kept so the project stays
honest about its origin. It runs through OpenCV's DNN module, so no
TensorFlow or PyTorch runtime is needed — the pipeline works anywhere
OpenCV does, which is why it is still the default.

It is fast rather than accurate. It struggles with small, blurred and
partly occluded people. YOLOv8 is available as an alternative backend
(`pip install ultralytics`), though a benchmark on this repository's ground
truth was inconclusive — see [docs/FINDINGS.md](../docs/FINDINGS.md) #6.

## The label file

The original notebook read `labels.txt` while the file on disk was named
`lable.txt`, so it raised `FileNotFoundError` as written. Renamed here to
`coco_labels.txt` and loaded through `humandetect/detector.py`.

# Models

| File | Committed | What it is |
|---|---|---|
| `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` | yes, 180 KB | Network architecture, read by OpenCV |
| `coco_labels.txt` | yes, 4 KB | The 80 COCO class names, in order. `person` is line 1. |
| `frozen_inference_graph.pb` | **no**, 13 MB | SSD MobileNet v3 trained weights |
| `yolov8n.pt` | **no**, 6.5 MB | YOLOv8n weights, fetched by ultralytics |

Neither weights file is committed. Both download automatically the first
time you run anything that needs them — the YOLO backend moves its weights
here rather than leaving them in whatever directory the process started
from. To fetch the SSD graph explicitly:

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
partly occluded people, and its 320x320 input is fixed by the frozen graph
— which turned out to be its central limitation.

An early comparison against YOLOv8n on this repository's own labelled
frames was inconclusive (finding #6). MOT17 settled it: on the benchmark's
7 train scenes, SSD scores HOTA 15.62 against YOLOv8n's 41.99, and the gap
is mostly false positives rather than misses — see findings #11 and #13.
YOLOv8n is what the headline result uses.

SSD remains the default in `cli.py` for two reasons that survive that
result: it needs no deep-learning runtime, and it carries no AGPL
obligation. Install `ultralytics` and pass `--backend yolo` to switch.

## The label file

The original notebook read `labels.txt` while the file on disk was named
`lable.txt`, so it raised `FileNotFoundError` as written. Renamed here to
`coco_labels.txt` and loaded through `humandetect/detector.py`.

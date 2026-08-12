#!/usr/bin/env bash
# Fetch the SSD MobileNet v3 Large (COCO) weights.
#
# The frozen graph is ~13 MB, so it is not committed. The .pbtxt config and
# the COCO label list are small and live in the repository.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p models

ARCHIVE="ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz"
URL="http://download.tensorflow.org/models/object_detection/${ARCHIVE}"

if [ -f models/frozen_inference_graph.pb ]; then
  echo "models/frozen_inference_graph.pb already present — nothing to do."
  exit 0
fi

echo "Downloading ${ARCHIVE}…"
curl -fL --progress-bar "$URL" -o "models/${ARCHIVE}"

echo "Extracting frozen graph…"
tar -xzf "models/${ARCHIVE}" -C models \
  --strip-components=1 \
  "ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb"

rm -f "models/${ARCHIVE}"
echo "Done: models/frozen_inference_graph.pb"

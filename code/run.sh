#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
[ -f data/teddy/depth.txt ] || { mkdir -p data/teddy && \
curl -L https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz \
| tar xz --strip-components=1 -C data/teddy; }
python code/main.py
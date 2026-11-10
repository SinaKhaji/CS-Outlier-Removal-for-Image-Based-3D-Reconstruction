#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

DATASET=scarecrow
OUT=data/out

curl -L -o data/$DATASET.zip https://igl.ethz.ch/projects/light-field-segmentation/light-field-segmentation-data/$DATASET.zip
unzip -q data/$DATASET.zip -d data/$DATASET
IMG=$(dirname "$(find data/$DATASET -iname '*.jpg' -o -iname '*.png' | head -1)")

mkdir -p $OUT
colmap feature_extractor  --database_path $OUT/db.db --image_path $IMG
colmap exhaustive_matcher --database_path $OUT/db.db
colmap mapper             --database_path $OUT/db.db --image_path $IMG --output_path $OUT
colmap image_undistorter  --image_path $IMG --input_path $OUT/0 --output_path $OUT/mvs --max_image_size 640
colmap patch_match_stereo --workspace_path $OUT/mvs --PatchMatchStereo.geom_consistency false --PatchMatchStereo.filter false
colmap stereo_fusion      --workspace_path $OUT/mvs --input_type photometric --output_path $OUT/fused.ply --StereoFusion.min_num_pixels 1

python3 code/main.py
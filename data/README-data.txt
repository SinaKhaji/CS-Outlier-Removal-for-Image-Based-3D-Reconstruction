https://igl.ethz.ch/projects/light-field-segmentation/
Download the datasets below:
scarecrow
decoration
dragon
statue
torch

run the below for each dataset
colmap feature_extractor --database_path out-scarecrow/db.db --image_path scarecrow
colmap exhaustive_matcher --database_path out-scarecrow/db.db
colmap mapper --database_path out-scarecrow/db.db --image_path scarecrow --output_path out-scarecrow
colmap image_undistorter --image_path scarecrow --input_path out-scarecrow/0 --output_path out-scarecrow/mvs --output_type COLMAP --max_image_size 640
colmap patch_match_stereo --workspace_path out-scarecrow/mvs --workspace_format COLMAP --PatchMatchStereo.geom_consistency false --PatchMatchStereo.filter false
colmap stereo_fusion --workspace_path out-scarecrow/mvs --workspace_format COLMAP --input_type photometric --output_path out-scarecrow/fused.ply --StereoFusion.min_num_pixels 1
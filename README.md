# Semantic-Segmentation-using-Implicit-Representation
## Getting Started
Modify the locations of the `locations_constant.py` 
..
## Useful commands
```
python main.py --frame_skip <frame_skip> --compressed <True|False> --show_only <True|False>
```
For debugging it is useful to turn on compressed (Will show it in 128x128 resolution) and show_only (will only visualize the iamges without exporting)

```
python visualize_gt_png.py -s <scene_name> -p <pose_id>
```
Visualizes the groundtruth png using SCANNET_200 

## Expected Datastructure
```
|
|
|-Semantic-Segmentation
    |-README.MD
    |-REST OF CODEBASE
|-PeRFception-ScanNet
    |-plenoxel_scannet_sceneXXXX_XX
        |-sample_renders
            |-pose_id.png
|-ScanNet
    |-scans
        |-sceneXXXX_XX
            |-pose
            |-intrinsic
```


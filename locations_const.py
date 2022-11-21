perfception_scannet_folder = "../PeRFception-ScanNet"
original_scannet_folder = "../ScanNet/scans"
scene_name = "scene0000_00"
plenoxel_prefix = "plenoxel_scannet_"
img_dir = "render_model"
semseg_poly_affix = "_vh_clean_2.labels.ply"
img_name = "image000.jpg"
render_dir = "./images"
cam_params = {
    "fx":1170.187988,
    "fy":1170.187988,
    "mx":647.750000,
    "my":483.750000
} # Color
img_params = {
    "width":1296,
    "height":968
} ## Probabbly high res. Low res might be 640, 480
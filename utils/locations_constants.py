perfception_scannet_folder = "../PeRFception-ScanNet"
original_scannet_folder = "../ScanNet/scans"
plenoxel_prefix = "plenoxel_scannet_"
img_dir = "render_model"
semseg_poly_affix = "_vh_clean_2.labels.ply"
img_name = "image000.jpg"
render_dir = "./images"
cam_params = {
    "fx":1169.621094,
    "fy":1167.105103,
    "mx":646.295044,
    "my":489.927032
} # Color
img_params = {
    "width":1296,
    "height":968
} ## Probabbly high res. Low res might be 640, 480
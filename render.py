import os
import numpy as np
import open3d as o3d
from read_scene import Scene
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.io.image import read_image
from torchvision.transforms.functional import to_pil_image
from scannet_constants import scannet_classes
import torch


print_separator = "==========="

def normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())

def vis_vox(links_idx, density):
    """
    Visualizes the voxels using O3D
    Uses density as colors
    """
    pts = links_idx.numpy().astype(np.float64)
    pts_color = (density - density.min()) / (density.max() - density.min())
    pts_color = pts_color.numpy().astype(np.float64).repeat(3, axis=-1)
    pts = np.concatenate([pts], axis=0)     
    pts_color = np.concatenate([pts_color], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(pts_color)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    perfception_scannet_folder = "../PeRFception-ScanNet"
    original_scannet_folder = "../ScanNet/scans"
    scene_name = "plenoxel_scannet_scene0000_00"
    img_dir = "render_model"
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
    scene = Scene(
        perf_folder= perfception_scannet_folder,
        orig_scannet_folder= original_scannet_folder,
        scene_name=scene_name
    )
    img_concat = os.path.join(scene.perf_folder,scene_name, img_dir, img_name)
    seg(image_path=img_concat)
    # vis_vox(links_idx= scene.links_idx, density=scene.density)


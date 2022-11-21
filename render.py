import os
import numpy as np
import open3d as o3d
from read_scene import Scene
from locations_const import *


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
    pcd.estimate_normals()
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 2 * avg_dist   
    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
    o3d.visualization.draw_geometries([mesh_o3d])
    

if __name__ == "__main__":
    scene = Scene(
        perf_folder= perfception_scannet_folder,
        orig_scannet_folder= original_scannet_folder,
        scene_name=scene_name
    )
    pts = scene.links_idx.numpy().astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    normals = pcd.estimate_normals()



    img_concat = os.path.join(scene.perf_folder,scene_name, img_dir, img_name)
    vis_vox(links_idx= scene.links_idx, density=scene.density)


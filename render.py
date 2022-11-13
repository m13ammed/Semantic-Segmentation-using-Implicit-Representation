import os
import json
import numpy as np
import open3d as o3d
import torch

print_separator = "==========="


def normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())

def debug_ckpt(ckpt):
    print(print_separator)
    print("Printing CKPT")
    for key in ckpt.keys():
        print(key)
        print(ckpt[key])
    print(print_separator)
    print("Printing State_Dict")
    for key in ckpt['state_dict'].keys():
        print(key)
        print(ckpt['state_dict'][key])
    print(print_separator)
    print("Printing SH")
    print("Length: ", len(ckpt['state_dict']['model.sh_data']))
    print(ckpt['state_dict']['model.sh_data'][0])
    print(print_separator)

def inspect_init(init):
    print(print_separator)
    print("Printing init")
    print("Length: ",len(init))
    print(init)
    print(print_separator)
    
def inspect_thick(thick):
    print(print_separator)
    print("Printing -thick-")
    print("Length: ", len(thick))
    print(thick)
    print(print_separator)

def inspect_trans(trans_info):
    print(print_separator)
    print("Printing trans info")
    for file in trans_info.files:
        print(file)
        print(trans_info[file])
    print(print_separator)
   

def vis_vox(ckpt, visualize=True):
    density = ckpt["state_dict"]["model.density_data"].detach().cpu()
    links_idx = ckpt["state_dict"]["model.links_idx"].detach().cpu()
    sh_data = ckpt['state_dict']['model.sh_data'].detach().cpu()
    valid = torch.where(density > 0.0)[0].long()
    density, links_idx = density[valid], links_idx[valid].long()
    sh_data = sh_data[valid].numpy().astype(np.float64)

    resolution = (
        ckpt["reso_list"]["reso_idx"] 
        if "reso_list" in ckpt.keys() else 
        [256, 256, 256]
    )
    
    links_idx = torch.stack(
        [
            links_idx // (resolution[1] * resolution[2]),
            links_idx % (resolution[1] * resolution[2]) // resolution[2],
            -links_idx % resolution[2],
        ],
        -1,
    )
    
    if(visualize):
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
    scene_folder = "/Users/kghandour/development/PeRFception-ScanNet/plenoxel_scannet_scene0000_00/"
    
    ## ckpt
    ckpt_path = os.path.join(scene_folder, "last.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(scene_folder, "data.ckpt")
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    debug_ckpt(ckpt=ckpt)

    ## 
    init_path = os.path.join(scene_folder, "init.npy")
    init = np.load(init_path)

    ##     
    thick_path = os.path.join(scene_folder, "thick.npy")
    thick = np.load(thick_path)

    ##
    trans_info_path = os.path.join(scene_folder,"trans_info.npz")
    trans_info = np.load(trans_info_path)


    vis_vox(ckpt=ckpt, visualize=True)

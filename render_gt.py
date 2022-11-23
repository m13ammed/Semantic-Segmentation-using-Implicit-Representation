from locations_const import *
import os
import torch
from generate_gt_renders import generate_gt_renders
from PIL import Image
import numpy as np
import glob

from read_scene import Scene


def export_images(target_images, show_only=False):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for idx in range(target_images.shape[0]):
        image = (clamp_and_detach(target_images[idx, ..., :3]))
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        img = Image.fromarray(tensor)
        if(show_only):
            img.show()
        else:
            img.save(os.path.join("./out/",str(idx)+".png"))

def find_files(dir, exts):
    if os.path.isdir(dir):
        files_grabbed = []
        for ext in exts:
            files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
        if len(files_grabbed) > 0:
            files_grabbed = sorted(files_grabbed)
        return files_grabbed
    else:
        return []

if __name__ == "__main__":
    pose_folder = os.path.join(perfception_scannet_folder,plenoxel_prefix+scene_name, "pose")
    scannet_scene = os.path.join(original_scannet_folder, scene_name)
    polygon_file = os.path.join(scannet_scene, scene_name+semseg_poly_affix)
    pose_files = find_files(pose_folder, exts=["*.txt"])
    assert len(pose_files) > 0, f"{pose_folder} does not contain poses"
    frame_ids = sorted([os.path.basename(f).rstrip(".txt") for f in pose_files])
    # scene = Scene(perf_folder=perfception_scannet_folder, orig_scannet_folder= original_scannet_folder, scene_name=plenoxel_prefix+scene_name)

    print(f"loading poses - {len(frame_ids)}")
    poses = np.stack(
        [np.loadtxt(os.path.join(pose_folder, f"{f}.txt")) for f in frame_ids],
        axis=0,
    )
    trans_info = np.load(os.path.join(perfception_scannet_folder,plenoxel_prefix+scene_name, "trans_info.npz"))

    poses = poses.astype(np.float32)

    # poses[:, :3, 3] *= scene_scale
    ## zero mean [end]
    # numerics = np.all(
    #     (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
    #     axis=1,
    # )
    # frame_ids = frame_ids[numerics]
    # poses = poses[numerics]

    # load intrinsics
    print(f"loading intrinsic")
    intrinsic = np.loadtxt(os.path.join(perfception_scannet_folder,plenoxel_prefix+scene_name, "intrinsic", "intrinsic_color.txt"))
    intrinsic = intrinsic.astype(np.float32)
    # intrinsic[[2, 3], [2, 3]] = 1
    intrinsic[2,2]=0
    intrinsic[2,3]=1
    intrinsic[3,2]=1
    intrinsic[3,3]=0

    # load trans_info

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")
    
    cameras, target_images = generate_gt_renders(poses=poses, polygon_path=polygon_file, device=device, intrinsic=intrinsic)
    export_images(target_images=target_images, show_only = True)



import glob
import os
import struct
import zlib

import cv2
import imageio
import numpy as np
from tqdm import tqdm

from dataloader.data_util.common import (
    connected_component_filter,
    find_files,
    similarity_from_cameras,
)







def load_plenoxel_scannet_data(
    datadir,
    cam_scale_factor=1.0,
    frame_skip=1,
    max_frame=1000,
    max_image_dim=800,
):
    files = find_files(os.path.join(datadir, "pose"), exts=["*.txt"])
    assert len(files) > 0, f"{datadir} does not contain color images."
    frame_ids = sorted([os.path.basename(f).rstrip(".txt") for f in files])

    num_frames = len(frame_ids)
    frames_in_use = (
        np.array(
            [np.floor(num_frames * (i / max_frame)) for i in range(max_frame)],
            dtype=np.int,
        )
        if max_frame != -1
        else np.arange(num_frames)
    )
    frames_in_use = np.unique(frames_in_use)
    
    frame_ids = np.array(frame_ids)[frames_in_use][::frame_skip]
    print("frames in use:", frame_ids)
    # prepare
    image = cv2.imread(os.path.join(datadir, "color", f"{frame_ids[0]}.jpg"))
    H, W = 968.0, 1296.0
    max_hw = max(H, W)
    resize_scale = max_image_dim / max_hw

    # load poses
    print(f"loading poses - {len(frame_ids)}")
    poses = np.stack(
        [np.loadtxt(os.path.join(datadir, "pose", f"{f}.txt")) for f in frame_ids],
        axis=0,
    )
    poses = poses.astype(np.float32)
    numerics = np.all(
        (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
        axis=1,
    )
    frame_ids = frame_ids[numerics]
    poses = poses[numerics]

    # load intrinsics
    print(f"loading intrinsic")
    intrinsic = np.loadtxt(os.path.join(datadir, "intrinsic", "intrinsic_color.txt"))
    intrinsic = intrinsic.astype(np.float32)
    intrinsic *= resize_scale
    intrinsic[[2, 3], [2, 3]] = 1

    # load trans_info
    trans_info = np.load(os.path.join(datadir, "trans_info.npz"))

    pcd_data = np.load(os.path.join(datadir, 'init.npy'))
    
    ## normalize
    #T, _ = similarity_from_cameras(poses)
    
    T = trans_info['T']
    poses = T @ poses
    ## normalize [end]

    ## zero mean
    pcd_mean = trans_info['pcd_mean'] #load
    #pcd_depth -= pcd_mean 
    poses[:, :3, 3] -= pcd_mean
    ## zero mean [end]

    
    scene_scale = trans_info['scene_scale']
    poses[:, :3, 3] *= scene_scale

    #####

    H, W = 480, 640
    i_split = np.arange(len(poses))
    i_test = np.unique(np.array([int(i * (len(poses) / 20)) for i in range(20)]))
    i_train = np.array([i for i in i_split if not i in i_test])
    print(f">> train: {len(i_train)}, test: {len(i_test)}, total: {len(i_split)}")
    render_poses = poses

    store_dict = {
        "poses": poses,
        "T": T,
        "scene_scale": scene_scale,
        "pcd_mean": pcd_mean,
        "pcd_voxel": pcd_data,
        "frame_ids": frame_ids,
        "class_info": None,
    }
    
    return (
        poses,
        render_poses,
        (int(H), int(W)),
        intrinsic,
        (i_train, i_test, i_test),
        store_dict,
    )

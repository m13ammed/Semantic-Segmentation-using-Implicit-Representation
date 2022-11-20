

import glob
import os
import struct
import zlib

import cv2
import imageio
import numpy as np
from tqdm import tqdm
from dataloader.interface import LitData

import gin

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

    #pcd_data = np.load(os.path.join(datadir, 'init.npy'))
    

    
    T = trans_info['T']
    poses = T @ poses


    ## zero mean
    pcd_mean = trans_info['pcd_mean'] #load
    #pcd_depth -= pcd_mean 
    poses[:, :3, 3] -= pcd_mean
    ## zero mean [end]

    
    scene_scale = trans_info['scene_scale']
    poses[:, :3, 3] *= scene_scale

    #####

    H, W = int(resize_scale*H), int(resize_scale*W) #480, 640
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
        #"pcd_voxel": pcd_data,
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

@gin.configurable()
class LitDataPefceptionScannet(LitData):
    def __init__(
        self,
        datadir: str,
        scene_name: str,
        accelerator: bool,
        num_gpus: int,
        num_tpus: int,
        # scannet specific arguments
        frame_skip: int = 1,
        max_frame: int = 1500,
        max_image_dim: int = 800,
        cam_scale_factor: float = 1.50,
    ):
        super(LitDataPefceptionScannet, self).__init__(
            datadir=datadir,
            accelerator=accelerator,
            num_gpus=num_gpus,
            num_tpus=num_tpus,
        )

        (
            extrinsics,
            render_poses,
            (h, w),
            intrinsics,
            i_split,
            trans_info,
        ) = load_plenoxel_scannet_data(
            os.path.join(datadir, scene_name),
            cam_scale_factor=cam_scale_factor,
            frame_skip=frame_skip,
            max_frame=max_frame,
            max_image_dim=max_image_dim,
        )
        i_train, i_val, i_test = i_split

        print(f"loaded scannet, image with size: {h} * {w}")
        self.scene_name = scene_name
        #self.images = images
        self.intrinsics = intrinsics.reshape(-1, 4, 4).repeat(len(extrinsics), axis=0)
        self.extrinsics = extrinsics
        self.image_sizes = np.array([h, w]).reshape(1, 2).repeat(len(extrinsics), axis=0)
        self.near = 0.0
        self.far = 1.0
        self.ndc_coeffs = (-1.0, -1.0)
        self.i_train, self.i_val, self.i_test = i_train, i_val, i_test
        self.i_all = np.arange(len(extrinsics))
        self.render_poses = render_poses
        self.trans_info = trans_info
        self.use_sphere_bound = False



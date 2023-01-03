

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

def round( n ):
    # Smaller multiple
    a = (n // 10) * 10
    # Larger multiple
    b = a + 10
    # Return of closest of two
    return (b if n - a > b - n else a)
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
    scene_name,
    cam_scale_factor=1.0,
    frame_skip=1,
    max_frame=-1,
    max_image_dim=800,
    scannet_dir = None,
    square = False):
    
    root_data_dir = datadir
    datadir = os.path.join(datadir,scene_name)
    perfception_prefix = 'plenoxel_scannet_'#_vh_clean_2.labels.ply       
    if scannet_dir is None:
        data_path_scan = datadir
        polygon_path = os.path.join(datadir,scene_name[len(perfception_prefix):]+'_vh_clean_2.labels.ply')
    else:
        polygon_path = os.path.join(scannet_dir,scene_name[len(perfception_prefix):],scene_name[len(perfception_prefix):]+ '_vh_clean_2.labels.ply')
        data_path_scan = os.path.join(scannet_dir,scene_name[len(perfception_prefix):])

    
    files = find_files(os.path.join(data_path_scan, "pose"), exts=["*.txt"])
    assert len(files) > 0, f"{data_path_scan} does not contain poses."
    frame_ids = [int(os.path.basename(f).rstrip(".txt")) for f in files]
    frame_ids = sorted(frame_ids)
    num_frames = len(frame_ids)
    frames_in_use = np.arange(min(max_frame*frame_skip,len(frame_ids))) if max_frame != -1 else np.arange(num_frames)
    
    
    frame_ids = np.array(frame_ids)[frames_in_use][::frame_skip]
    print("frames in use:", frame_ids, frame_skip)
    # prepare
    #image = cv2.imread(os.path.join(datadir, "color", f"{frame_ids[0]}.jpg"))
    H, W = 968.0, 1296.0
    max_hw = max(H, W)
    
    max_image_dim = round(max_image_dim) if not square else max_image_dim

    resize_scale = [max_image_dim/H, max_image_dim/W] if square else max_image_dim / max_hw

    # load poses
    print(f"loading poses - {len(frame_ids)}")
    poses = np.stack(
        [np.loadtxt(os.path.join(data_path_scan, "pose", f"{f}.txt")) for f in frame_ids],
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
    intrinsic = np.loadtxt(os.path.join(data_path_scan, "intrinsic", "intrinsic_color.txt"))
    intrinsic = intrinsic.astype(np.float32)
    intrinsic_orig = intrinsic.copy()
    if square:
        intrinsic[0,:] *= resize_scale[1]
        intrinsic[1,:] *= resize_scale[0]
    else:
        intrinsic *= resize_scale
    intrinsic[[2, 3], [2, 3]] = 1
    intrinsic_orig[2,2]=0
    intrinsic_orig[2,3]=1
    intrinsic_orig[3,2]=1
    intrinsic_orig[3,3]=0

    
    # load trans_info
    trans_info = np.load(os.path.join(datadir, "trans_info.npz"))

    #pcd_data = np.load(os.path.join(datadir, 'init.npy'))
    

    
    T = trans_info['T']
    render_poses = T @ poses


    ## zero mean
    pcd_mean = trans_info['pcd_mean'] #load
    #pcd_depth -= pcd_mean 
    render_poses[:, :3, 3] -= pcd_mean
    ## zero mean [end]

    
    scene_scale = trans_info['scene_scale']
    render_poses[:, :3, 3] *= scene_scale

    #####
    if square:

        H, W = (int(max_image_dim), int(max_image_dim))
    else: 
        H, W = int(round(resize_scale*H)), round(int(resize_scale*W)) #480, 640
    i_split = np.arange(len(render_poses))
    i_test = np.unique(np.array([int(i * (len(render_poses) / 20)) for i in range(20)]))
    i_train = np.array([i for i in i_split if not i in i_test])
    print(f">> train: {len(i_train)}, test: {len(i_test)}, total: {len(i_split)}")
    
    store_dict = {
        "poses": poses,
        "T": T,
        "scene_scale": scene_scale,
        "pcd_mean": pcd_mean,
        #"pcd_voxel": pcd_data,
        "frame_ids": frame_ids,
        "class_info": None,
        "polygon": polygon_path,
        "intrinsic_orig":intrinsic_orig
    }
    
    return (
        poses,
        render_poses,
        (int(H), int(W)),
        intrinsic,
        (i_train, i_test, i_test),
        polygon_path,
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
        max_frame: int = -1,
        max_image_dim: int = 800,
        cam_scale_factor: float = 1.50,
        scannet_dir = None,
        square = False
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
            polygon,
            trans_info,
        ) = load_plenoxel_scannet_data(
            datadir,
            scene_name = scene_name,
            cam_scale_factor=cam_scale_factor,
            frame_skip=frame_skip,
            max_frame=max_frame,
            max_image_dim=max_image_dim,
            scannet_dir=scannet_dir,
            square= square
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
        self.polygon = polygon
        self.intrinsic_orig = trans_info["intrinsic_orig"].reshape(-1, 4, 4).repeat(len(extrinsics), axis=0)



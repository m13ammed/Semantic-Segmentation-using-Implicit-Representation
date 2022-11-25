import os, glob
import numpy as np
from utils.locations_constants import *
from utils.scannet_scene import ScanNetScene

class DataLoader:
    """
    Loads the ScanNet folder
    """
    def find_files(self,dir, exts=None):
        if os.path.isdir(dir):
            files_grabbed = []
            for ext in exts:
                files_grabbed.extend(glob.glob(os.path.join(dir, ext)))
            if len(files_grabbed) > 0:
                files_grabbed = sorted(files_grabbed, key=len)
            return files_grabbed
        else:
            return []

    def __init__(
        self,
        cam_scale_factor=1.0,
        frame_skip=20,
        max_frame=1000,
        max_image_dim=800,
        scene_name=None
    ):
        self.scannet_scenes = []            
        scene_names = sorted(glob.glob(original_scannet_folder+"/*/",recursive=True))
        for scene_idx in scene_names:
            pose_files = self.find_files(os.path.join(scene_idx, "pose"), exts=["*.txt"])
            if(len(pose_files)==0):
                scene_names.remove(scene_idx)
                continue
            frame_ids = sorted([os.path.basename(f).rstrip(".txt") for f in pose_files], key=lambda name: int(name))
            num_frames = len(frame_ids)
            # frames_in_use = (
            #     np.array(
            #         [np.floor(num_frames * (i / max_frame)) for i in range(max_frame)],
            #         dtype=np.int,
            #     )
            #     if max_frame != -1
            #     else np.arange(num_frames)
            # )
            # frames_in_use = np.unique(frames_in_use)
            frame_ids = np.array(frame_ids)[::frame_skip]
            print(f"Loading Scene - {scene_idx}")
            print(f"loading poses - {len(frame_ids)}")
            poses = np.stack(
                [np.loadtxt(os.path.join(scene_idx, "pose", f"{f}.txt")) for f in frame_ids],
                axis=0,
            )
            poses = poses.astype(np.float32)
            # numerics = np.all(
            #     (~np.isinf(poses) * ~np.isnan(poses) * ~np.isneginf(poses)).reshape(-1, 16),
            #     axis=1,
            # )
            # frame_ids = frame_ids[numerics]
            # poses = poses[numerics]

            # load intrinsics
            print(f"loading intrinsic")
            intrinsic = np.loadtxt(os.path.join(scene_idx, "intrinsic", "intrinsic_color.txt"))
            intrinsic = intrinsic.astype(np.float32)
            intrinsic[2,2]=0
            intrinsic[2,3]=1
            intrinsic[3,2]=1
            intrinsic[3,3]=0

            scene_stripped = scene_idx.split("/")
            print(f"loading polygon")
            polygon = os.path.join(scene_idx,str(scene_stripped[-2])+str(semseg_poly_affix))


            scannet_scene = ScanNetScene(scene_name=scene_idx, pose_ids=frame_ids, poses=poses, intrinsics=intrinsic, polygon=polygon)
            self.scannet_scenes.append(scannet_scene)


    def get_scannet_scenes(self):
        return self.scannet_scenes
    

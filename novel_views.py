from PIL import Image
import numpy as np
import glob
import os
from configs.scannet_constants import *
from tqdm import tqdm
import argparse
import pandas as pd
import math
path_depth = "/home/rozenberszki/Downloads/ScanNet-gt-124-depth"
analysis_dict_path = "/home/rozenberszki/Downloads/class_dist_center.npy"
scannet_path = "/home/rozenberszki/Downloads/ScanNet/scans"

def process_pose(cls_id, pose_path, depth_path, seg_path):
    pose_np = np.loadtxt(pose_path)
    print(depth_path)
    print(seg_path)
    depth = np.load(depth_path, allow_pickle='TRUE')
    seg = np.load(seg_path, allow_pickle='TRUE')
    seg_idx = np.argwhere(seg==cls_id)
    idx_example = seg_idx[0]
    depth_at_idx = depth[idx_example[0], idx_example[1]]
    print(depth_at_idx)
    print(pose_np, pose_np[1,0], pose_np[0,0])
    angle_z = np.degrees(np.arctan2(pose_np[1,0], pose_np[0,0]))
    trans_x = depth_at_idx * np.cos(angle_z)
    print(depth_at_idx, trans_x)
    exit()

if __name__=="__main__":    
    analysis_dict = np.load(analysis_dict_path, allow_pickle='TRUE').item()
    least_4_classes = [36, 34, 11, 28]
    cls_dict = {}
    for i in least_4_classes:
        val = VALID_CLASS_IDS_20.index(i)+1
        cls_dict[val] = analysis_dict[i]

    for cls_id in tqdm(cls_dict.keys()):
        cls = cls_dict[cls_id]
        for scene_id in tqdm(cls['scenes'].keys()):
            scannet_scene_id = '_'.join(scene_id.split("_")[2:])
            scene = cls['scenes'][scene_id]
            pose_dir = os.path.join(scannet_path, scannet_scene_id, "pose")
            for pose in tqdm(scene['poses']):
                pose_id = pose.split(".")[0]
                pose_path = os.path.join(pose_dir, pose_id+".txt")
                depth_path = os.path.join(path_depth, scene_id, pose_id+"_depth.npy")
                seg_path = os.path.join(path_depth, scene_id, pose_id+".npy")
                process_pose(cls_id=cls_id, pose_path=pose_path, depth_path=depth_path, seg_path=seg_path)
    
    # for folder in tqdm(sorted(os.listdir(path_depth))):
    #     scene_path = os.path.join(path_depth, folder)
    #     for file in sorted(os.listdir(scene_path)):
    #         ## Get depth
    #         if(not file.endswith("depth.npy")):
    #             continue
    #         print(file)
    #     exit()
        
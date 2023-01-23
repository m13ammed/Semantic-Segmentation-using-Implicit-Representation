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


# def get_target_point(pose, depth, angles):
#     point = pose[:3,3:]
#     translate_matrix = np.eye(4)
#     translate_matrix[:3,3:] = -point

#     # Rotate point
#     angle_rad = angles[0]
#     rotate_matrix = [
#         [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
#         [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ]
#     # point[1] += depth

#     angle_rad = -angles[0]
#     rotate_back_matrix = [
#         [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
#         [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
#         [0, 0, 1, 0],
#         [0, 0, 0, 1]
#     ]
    
#     translate_back_matrix = np.eye(4)
#     translate_back_matrix[:3, 3:] = point
#     transformation_matrix = np.zeros(4)

#     transformation_matrix = translate_back_matrix @ rotate_back_matrix

#     transformation_matrix[0, 3]+=depth
#     transformation_matrix = transformation_matrix @ rotate_matrix

#     transformation_matrix = transformation_matrix @ translate_matrix
    
#     return transformation_matrix @ pose

def get_target_point(pose, depth, angles):
    target = pose.copy()
    pitch = angles[0]
    yaw = angles[1]
    Py = depth * math.sin(yaw) * math.cos(pitch) 
    Px = depth * math.cos(yaw) 
    Pz = depth * math.cos(yaw) * math.cos(pitch)
    target[0,3] += Px
    target[1,3] += Py
    # target[1,3] += depth * math.cos(angles[1])
    return target

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    R = R[:3, :3]
 
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])

def rotate_point_matrix(init_pose, angle, rot_point):
    # Translate point to origin
    point = rot_point.copy()[:3, 3:]
    # point = init_pose.copy()[:3, 3:]
    translate_matrix = np.eye(4)
    translate_matrix[:3,3:] = -point

    # Rotate point
    angle_rad = math.radians(angle)

    rotate_matrix_z = [
        [math.cos(angle_rad), -math.sin(angle_rad), 0, 0],
        [math.sin(angle_rad), math.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    rotate_matrix_x = [
        [1, 0, 0, 0],
        [0, math.cos(angle_rad), -math.sin(angle_rad), 0],
        [0, math.sin(angle_rad), math.cos(angle_rad), 0],
        [0,0,0,1]
    ]
    
    rotate_matrix_y = [
        [math.cos(angle_rad), 0, math.sin(angle_rad), 0],
        [0, 1, 0, 0],
        [-math.sin(angle_rad), 0, math.cos(angle_rad), 0],
        [0,0,0,1]
    ]
    
    translate_back_matrix = np.eye(4)
    translate_back_matrix[:3, 3:] = point
    transformation_matrix = np.zeros(4)

    transformation_matrix = translate_back_matrix @ rotate_matrix_z
    transformation_matrix = transformation_matrix @ translate_matrix
    
    return transformation_matrix @ init_pose
    
def create_dome(init_pose, angle, point, step_size):
    ret_arr = []
    for i in range(step_size, angle, step_size):
        new_pose = rotate_point_matrix(init_pose.copy(), i, point.copy())
        ret_arr.append(new_pose)
    for i in range(-step_size, -angle, -step_size):
        new_pose = rotate_point_matrix(init_pose.copy(), i, point.copy())
        ret_arr.append(new_pose)
    
    return ret_arr

def process_pose(cls_id, pose_path, depth_path, seg_path):
    pose_np = np.loadtxt(pose_path)
    depth = np.load(depth_path, allow_pickle='TRUE')
    seg = np.load(seg_path, allow_pickle='TRUE')
    seg_idx = np.argwhere(seg==cls_id)
    idx_example = seg_idx[0]
    depth_at_idx = depth[idx_example[0], idx_example[1]]
    angles = rotationMatrixToEulerAngles(pose_np)
    target_point = get_target_point(pose_np, depth_at_idx, angles)
    arr = create_dome(pose_np, 90, target_point, 15)
    # print("Initial \n", pose_np)
    # print("Target \n", target_point)
    # np.savetxt("test.txt", target_point)
    return arr, target_point

if __name__=="__main__":    
    new_poses_dir = "/home/rozenberszki/Downloads/New_Poses"
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
            for p_i, pose in enumerate(tqdm(scene['poses'])):
                pose_id = pose.split(".")[0]
                pose_path = os.path.join(pose_dir, pose_id+".txt")
                depth_path = os.path.join(path_depth, scene_id, pose_id+"_depth.npy")
                seg_path = os.path.join(path_depth, scene_id, pose_id+".npy")
                new_poses, target_pt = process_pose(cls_id=cls_id, pose_path=pose_path, depth_path=depth_path, seg_path=seg_path)
                for index, new_pose in enumerate(new_poses):
                    Save_path = os.path.join(new_poses_dir, scene_id,"pose")
                    Save_target_path = os.path.join(new_poses_dir, scene_id, "target_pt")
                    os.makedirs(Save_target_path, exist_ok=True)
                    Save_target_pt = os.path.join(Save_target_path, str(cls_id)+"_"+str(p_i)+"_"+str(index)+".txt")
                    os.makedirs(Save_path, exist_ok=True)
                    np.savetxt(Save_target_pt, target_pt)
                    save_file_path = os.path.join(Save_path, str(cls_id)+"_"+str(p_i)+"_"+str(index)+".txt")
                    np.savetxt(save_file_path, new_pose)


    # for folder in tqdm(sorted(os.listdir(path_depth))):
    #     scene_path = os.path.join(path_depth, folder)
    #     for file in sorted(os.listdir(scene_path)):
    #         ## Get depth
    #         if(not file.endswith("depth.npy")):
    #             continue
    #         print(file)
    #     exit()
        
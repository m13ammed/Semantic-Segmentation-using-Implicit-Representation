from cProfile import label
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import gin
import torch.utils.data as data
from utils.locations_constants import *
from glob import glob
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from configs.scannet_constants import SCANNET_COLOR_MAP_200, SCANNET_COLOR_MAP_20, remap_200_to_20
from pytorch3d.structures import join_meshes_as_batch
from configs.train import train_scenes
from configs.val import val_scenes
from configs.test import test_scenes
from utils.ply_load import load_mesh_labels
from torch.nn.utils.rnn import pad_sequence
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]
def custom_collate(data):

        
    rgb = torch.stack(tuple([d[0] for d in data]), 0)
    intrinsic = torch.stack(tuple([d[1] for d in data]), 0)
    poses = torch.stack(tuple([d[2] for d in data]), 0)
    mesh = join_meshes_as_batch(tuple([d[3] for d in data]))
    labels = pad_sequence(tuple([d[4] for d in data]), batch_first=True, padding_value = 0)
    
    if len(data[0]) == 6:
        color = pad_sequence(tuple([d[5] for d in data]), batch_first=True, padding_value = 0)
        return (rgb, intrinsic, poses, mesh, labels, color)
    else:
        return (rgb, intrinsic, poses, mesh, labels)

@gin.configurable()
class LitPerfception(data.Dataset):
    def __init__(
        self,
        perf_root='/data/',
        scannet_root = '/scannet_data/ScanNet/scans/',
        mode = "train",
        frame_skip = None, #Not Implemented,
        debug = False
    ):
        super().__init__()
        self.orgin_size = [968.0, 1296.0] #H, W
        self.color_mean, self.color_std  = [0.496342, 0.466664, 0.440796], [0.277856, 0.286230, 0.291129]
        self.normalize = transforms.Normalize(mean=self.color_mean, std=self.color_std)
        self.mode = mode
        perf_prefix = 'plenoxel_scannet_'
        self.intrinsics_scale = None
        self.debug = debug
        if mode == 'train':
            perf_scenes = train_scenes
        elif mode == "val":
            perf_scenes = val_scenes
        elif mode == 'test':
            perf_scenes = test_scenes
        #scannet_scenes = [f[len(perf_prefix):] for f in perf_scenes]
        rgb_images_paths = []
        perf_intrinsics_list = []
        poses_list = []
        meshes_list = []
        for scene in perf_scenes:
            rgb_paths = sorted(glob(os.path.join(perf_root,scene,'*.jpg')))
            rgb_images_paths = rgb_images_paths + rgb_paths
            perf_intrinsics_list.append(np.load(os.path.join(perf_root,scene,'intrinsics.npy')))
            poses_list.append(np.load(os.path.join(perf_root,scene,'poses.npy')))
            scannet_scene = scene[len(perf_prefix):]
            meshes_list_ = [os.path.join(scannet_root,scannet_scene, scannet_scene+'_vh_clean_2.labels.ply')] * len(rgb_paths)
            meshes_list = meshes_list + meshes_list_        
        self.rgb_images_paths = rgb_images_paths
        self.perf_intrinsics = np.concatenate(perf_intrinsics_list, axis=0)#.squeeze(0)
        self.poses = np.concatenate(poses_list, axis=0)#.squeeze(0)
        self.meshes_list = meshes_list
        assert len(self.meshes_list) == len(self.rgb_images_paths), "num of meshes not equal to num of images"
        assert self.poses.shape[0] == self.perf_intrinsics.shape[0], "num of poses not equal to num of perf_intrinsics"
        assert self.poses.shape[0] == len(self.rgb_images_paths), "num of poses not equal to num of meshes"
    def __len__(self):
        return len(self.rgb_images_paths)
    def __getitem__(self, index):
        rgb_path = self.rgb_images_paths[index]
        rgb = np.array(Image.open(rgb_path))
        if self.intrinsics_scale is None:
            self.intrinsics_scale = [self.orgin_size[0]/rgb.shape[0], self.orgin_size[1]/rgb.shape[1]]
            # max_image_dim/H inverting this
        # Reshape data from H x W x C to C x H x W
        rgb = np.moveaxis(rgb, 2, 0)
        # Define normalizing transform
        # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
        rgb = self.normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))        
        perf_intrinsics = self.perf_intrinsics[index]
        intrinsic = perf_intrinsics.copy()
        intrinsic[0,:] *= self.intrinsics_scale[1]
        intrinsic[1,:] *= self.intrinsics_scale[0]
        intrinsic[[2, 3], [2, 3]] = 1
        intrinsic = torch.Tensor(intrinsic.astype(np.float32))
        intrinsic[2,2]=0
        intrinsic[2,3]=1
        intrinsic[3,2]=1
        intrinsic[3,3]=0

        poses = self.poses[index]
        poses = torch.Tensor(poses.astype(np.float32))
        mesh_path = self.meshes_list[index]
        mesh, labels, color = load_mesh_labels(mesh_path, torch.device("cpu"))
        if self.debug:
            return rgb, intrinsic, poses, mesh, labels, color
        
        else:
            return rgb, intrinsic, poses, mesh, labels
        
class Perfception(data.Dataset):
    def __init__(
        self,
        perf_root = perfception_scannet_folder,
        scannet_root = original_scannet_folder,
        image_folder = sample_renders,
        seg_classes = 'SCANNET20',
        frame_skip = 20,
        mode = "train"
    ):
        ## TODO: Construct a coherent directory structure and simplify reading
        super().__init__()
        self.mode = mode
        self.length = 0

        perf_set = sorted(glob(perf_root+"/*/"))
        scannet_set = sorted(glob(scannet_root+"/*/"))
        if self.mode.lower() == 'train':
            self.train_data = []
            self.train_labels = []
            for scene in perf_set:
                color_images = glob(os.path.join(scene, sample_renders,"*.jpg"))
                self.train_data += color_images
                self.length += len(color_images)
            for scene in scannet_set:
                gt_labels = glob(os.path.join(scene, labels_folder, "*.png"))
                self.train_labels += gt_labels

        if self.mode.lower() == 'val':
            self.val_data = []
            self.val_labels = []
            for scene in perf_set:
                color_images = glob(os.path.join(scene, sample_renders,"*.jpg"))
                self.val_data += color_images
                self.length += len(color_images)
            for scene in scannet_set:
                gt_labels = glob(os.path.join(scene, labels_folder, "*.png"))
                self.val_labels += gt_labels
            
        if self.mode.lower() == 'test':
            self.test_data = []
            self.test_labels = []
            for scene in perf_set:
                color_images = glob(os.path.join(scene, sample_renders,"*.jpg"))
                self.test_data += color_images
                self.length += len(color_images)
            for scene in scannet_set:
                gt_labels = glob(os.path.join(scene, labels_folder, "*.png"))
                self.test_labels += gt_labels

    def __getitem__(self, index):
        if self.mode.lower() == 'train':
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == 'val':
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == 'test' or self.mode.lower() == 'inference':
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError('Unexpected dataset mode. Supported modes are: train, val, test')
        
        data = np.array(Image.open(data_path).resize((640,480))).astype(np.uint8)
        data = np.moveaxis(data, 2, 0)
        normalize = transforms.Normalize(mean=0, std=[1,1,1])
        data = normalize(torch.Tensor(data.astype(np.float32) / 255.0))
        label = np.array(Image.open(label_path).resize((640,480))).astype(np.uint8)
        for key in SCANNET_COLOR_MAP_200.keys():
            key_20 = remap_200_to_20(key=key)
            idx = label == key
            label[idx] = key_20
            label[label==13] = 0
            label[label==31] = 0
            label[label>40] = 0
        
        
        return data, label

    def __len__(self):
        return self.length

            


#d = LitPerfception(perf_root='/data/', scannet_root = '/scannet_data/ScanNet/scans/')
#
#
#train_loader = data.DataLoader(d,batch_size=10,shuffle=True, num_workers=0, collate_fn=custom_collate)
#
#for batch in train_loader:
#    
#    pass

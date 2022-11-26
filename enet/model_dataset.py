import torch.utils.data as data
from utils.locations_constants import *
from glob import glob
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import torch
from utils.scannet_constants import SCANNET_COLOR_MAP_200, SCANNET_COLOR_MAP_20, remap_200_to_20


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

            



        




        


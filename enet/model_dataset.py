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
from configs.train import train_scenes, train_scenes_opt, train_scenes_full
from configs.val import val_scenes, val_scenes_opt
from configs.test import test_scenes
from utils.ply_load import load_mesh_labels
from torch.nn.utils.rnn import pad_sequence
color_mean = [0.496342, 0.466664, 0.440796]
color_std = [0.277856, 0.286230, 0.291129]
color_new_mean = [0.50275189, 0.45223128, 0.39241619]
color_new_std = [0.28362849, 0.27880838, 0.27457628]
sh_mean = [ 0.00824186,  0.11539002,  0.06089742,  0.06155786,  0.05825128,
        0.0593638 ,  0.09168289,  0.05926862,  0.11279657, -0.04009821,
        0.14060101,  0.06198773,  0.0599228 ,  0.05789624,  0.05871703,
        0.08233076,  0.0594893 ,  0.09212159, -0.08556332,  0.1680364 ,
        0.06273864,  0.0581194 ,  0.05829075,  0.05739062,  0.06550323,
        0.05976876,  0.0617659 ]
sh_std = [0.36573967, 0.36615546, 0.31597748, 0.31306109, 0.50109065,
       0.48896791, 0.44744172, 0.38749937, 0.54356951, 0.36309892,
       0.35698321, 0.31197455, 0.3092862 , 0.49163777, 0.47873379,
       0.43714261, 0.37999565, 0.52787927, 0.36710315, 0.36049178,
       0.31901154, 0.31674267, 0.4975526 , 0.48383487, 0.44135966,
       0.38612645, 0.52761967]
def custom_collate(data):
    rgb, intrinsic, poses          = [], [], []
    mesh, labels, color            = [], [], []
    exists, labels_2d, sh          = [], [], []

    debug = 'color' in data[0].keys()
    gen_seg = 'poses' in data[0].keys()
    use_sh = 'sh' in data[0].keys()
    for i, d in enumerate(data):
        rgb.append(d['rgb'])
        labels_2d.append(d['label_2d'])
        if 'poses' in d.keys():
            intrinsic.append(d['intrinsic'])
            poses.append(d['poses'])
            mesh.append(d['mesh'])
            labels.append(d['labels'])
            if not d['sem_exists']:
                exists.append(i)
        if debug:
            color.append(d['color'])    
        if use_sh:
            sh.append(d['sh'])
    rgb = torch.stack(rgb, 0)
    labels_2d = torch.stack(labels_2d, 0)
    ret = {
        'rgb': rgb,
        'labels_2d': labels_2d
    }
    if gen_seg and len(exists):
        intrinsic = torch.stack(intrinsic, 0)
        poses = torch.stack(poses, 0)
        mesh = join_meshes_as_batch(mesh)
        labels = pad_sequence(labels, batch_first=True, padding_value = 0)
        ret.update({
            'intrinsic':intrinsic,
            'poses':poses,
            'mesh':mesh,
            'labels':labels,
            'idx_to_ren': exists
        })
    if debug:
        color = pad_sequence(color, batch_first=True, padding_value = 0)
        ret.update({
            'color': color
        })
    if use_sh:
        ret.update({
            'sh': torch.stack(sh, 0)
        })
    return ret
@gin.configurable()
class LitPerfception(data.Dataset):
    def __init__(
        self,
        perf_root='/data/',
        scannet_root = '/scannet_data/ScanNet/scans/',
        gt_seg_root = '/seg_data/', #path to already generated 
        allow_gen_lables = True, #wether to allow generating labels if seg image is not found
        use_sh = True,
        use_original_norm = False,
        mode = "train",
        frame_skip = None, #Not Implemented,
        debug = False,
        opt = False
        
    ):
        super().__init__()
        self.orgin_size = [968.0, 1296.0] #H, W
        if use_original_norm:
            self.color_mean, self.color_std  = [0.496342, 0.466664, 0.440796], [0.277856, 0.286230, 0.291129]
        else:
            self.color_mean, self.color_std  = color_new_mean, color_new_std
        self.normalize = transforms.Normalize(mean=self.color_mean, std=self.color_std)
        if use_sh:
            self.sh_mean, self.sh_std = np.array(sh_mean).reshape(-1,1,1), np.array(sh_std).reshape(-1,1,1)
        self.mode = mode
        perf_prefix = 'plenoxel_scannet_'
        self.intrinsics_scale = None
        self.debug = debug
        self.allow_gen_lables = allow_gen_lables
        self.use_sh = use_sh
        
        if opt:
            if mode == 'train':
                perf_scenes = train_scenes_opt
            elif mode == "val":
                perf_scenes = val_scenes_opt
        else:
            if mode == 'train':
                perf_scenes = train_scenes_full
            elif mode == "val":
                perf_scenes = test_scenes#val_scenes
            elif mode == 'test':
                perf_scenes = test_scenes
        #scannet_scenes = [f[len(perf_prefix):] for f in perf_scenes]
        rgb_images_paths = []
        seg_images_paths = []
        sh_paths = []
        perf_intrinsics_list = []
        poses_list = []
        meshes_list = []
        for scene in perf_scenes:
            rgb_paths = sorted(glob(os.path.join(perf_root,scene,'*.jpg')))
            rgb_images_paths = rgb_images_paths + rgb_paths
            if self.allow_gen_lables:
                perf_intrinsics_list.append(np.load(os.path.join(perf_root,scene,'intrinsics.npy')))
                poses_list.append(np.load(os.path.join(perf_root,scene,'poses.npy')))
                scannet_scene = scene[len(perf_prefix):]
                meshes_list_ = [os.path.join(scannet_root,scannet_scene, scannet_scene+'_vh_clean_2.labels.ply')] * len(rgb_paths)
                meshes_list = meshes_list + meshes_list_  
            #seg_images_paths.append(sorted(glob(os.path.join(gt_seg_root,scene,'*.jpg'))))
            seg_paths = [os.path.join(gt_seg_root,scene, str(int(i.split('/')[-1][5:-4]))+'.npy') for i in rgb_paths]
            seg_images_paths = seg_images_paths + seg_paths
            if use_sh:
                sh_paths = sh_paths +sorted(glob(os.path.join(perf_root,scene,'*.npz')))
        self.seg_image_exists = [os.path.exists(pth) for pth in seg_images_paths] 
        self.rgb_images_paths = rgb_images_paths
        self.sh_paths =  sh_paths
        self.seg_images_paths = seg_images_paths
        if self.allow_gen_lables:
            self.perf_intrinsics = np.concatenate(perf_intrinsics_list, axis=0)#.squeeze(0)
            self.poses = np.concatenate(poses_list, axis=0)#.squeeze(0)
            self.meshes_list = meshes_list
            assert len(self.meshes_list) == len(self.rgb_images_paths), "num of meshes not equal to num of images"
            assert self.poses.shape[0] == self.perf_intrinsics.shape[0], "num of poses not equal to num of perf_intrinsics"
            assert self.poses.shape[0] == len(self.rgb_images_paths), "num of poses not equal to num of meshes"
        assert len(self.rgb_images_paths) == len(self.seg_images_paths), "num of rgb not equal to num of possioble gen images"
        assert len(self.rgb_images_paths) == len(self.seg_image_exists), "num of rgb not equal to num of seg_image_exists"
        if not allow_gen_lables:
            assert (False not in self.seg_image_exists), "there are some not generated labels"
        if use_sh:
            assert len(self.rgb_images_paths) == len(self.sh_paths), "num of poses not equal to num of sh files"
    def __len__(self):
        return len(self.rgb_images_paths)
    def __getitem__(self, index):
        rgb_path = self.rgb_images_paths[index]
        rgb = np.array(Image.open(rgb_path))
        # Reshape data from H x W x C to C x H x W
        rgb = np.moveaxis(rgb, 2, 0)
        # Define normalizing transform
        # Convert image to float and map range from [0, 255] to [0.0, 1.0]. Then normalize
        rgb = self.normalize(torch.Tensor(rgb.astype(np.float32) / 255.0))        

        exists = self.seg_image_exists[index]
        if exists:
            #label_2d = torch.Tensor(np.array(Image.open(self.seg_images_paths[index])))
            label_2d = torch.Tensor(np.load(self.seg_images_paths[index]))
        else:
            label_2d = torch.zeros((rgb.shape[1],rgb.shape[2]))
        
        ret_dict = {'rgb':rgb,
                    'label_2d':label_2d,} #add semantics and 
        
        if self.use_sh:
            sh = np.load(self.sh_paths[index])['arr_0.npy']
            sh = np.moveaxis(sh, 2, 0)
            sh_mean = np.tile(self.sh_mean, (1,sh.shape[-2], sh.shape[-1]))
            sh_std = np.tile(self.sh_std, (1,sh.shape[-2], sh.shape[-1]))
            assert sh.shape == sh_mean.shape and sh.shape == sh_std.shape, f'Error in normalization shape {sh.shape}, {sh_mean.shape}, {sh_std.shape}'
            sh = (sh-sh_mean)/sh_std
            ret_dict.update({'sh':torch.Tensor(sh)})
        
        if self.debug:
            ret_dict.update({'color':color,})
        if self.allow_gen_lables and not exists:
            poses = self.poses[index]
            poses = torch.Tensor(poses.astype(np.float32))

            if self.intrinsics_scale is None:
                self.intrinsics_scale = [self.orgin_size[0]/rgb.shape[0], self.orgin_size[1]/rgb.shape[1]]
                # max_image_dim/H inverting this
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
            mesh_path = self.meshes_list[index]
            mesh, labels, color = load_mesh_labels(mesh_path, torch.device("cpu"))

            ret_dict.update({
            'intrinsic':intrinsic,
            'poses':poses,
            'mesh':mesh,
            'labels':labels,
            'sem_exists': exists,
        })      
        return ret_dict

        
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

            

#
#d = LitPerfception(perf_root='/data/', scannet_root = '/scannet_data/ScanNet/scans/')
#
#
#train_loader = data.DataLoader(d,batch_size=8,shuffle=False, num_workers=0, collate_fn=custom_collate)
#
#for batch in train_loader:
#    
#    print('xx')
#    break

    

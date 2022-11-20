from PIL import Image 
import os
from torchvision.transforms.functional import pil_to_tensor
import numpy as np
import sys
from scannet_constants import SCANNET_COLOR_MAP_200, VALID_CLASS_IDS_200
from read_scene import Scene

np.set_printoptions(threshold=sys.maxsize)


if __name__ == "__main__":
    perfception_scannet_folder = "../PeRFception-ScanNet"
    original_scannet_folder = "../ScanNet/scans"
    perfception_prefix = "plenoxel_scannet_"
    scene_name = "scene0000_00"
    perfception_img_dir = "render_model"
    scannet_img_dir = "label-filt"

    perfception_img_name = "image000.jpg"
    scannet_img_name = "0.png"

    scene = Scene(perf_folder=perfception_scannet_folder, orig_scannet_folder=original_scannet_folder, scene_name=perfception_prefix+scene_name)
    print(scene.trans_info['frame_ids'][16])

    perf_img_path = os.path.join(perfception_scannet_folder, perfception_prefix+scene_name, perfception_img_dir, perfception_img_name)
    scannet_img_path = os.path.join(original_scannet_folder, scene_name, scannet_img_dir, scannet_img_name)

    im_perf = Image.open(perf_img_path)
    im_scannet = Image.open(scannet_img_path).resize((640,480))

    im_perf.show()
    # im_scannet.show()

    non_segm = np.array(im_perf)
    gt_segm = np.array(im_scannet)
    print(non_segm.shape)
    for key in SCANNET_COLOR_MAP_200.keys():
        idx = gt_segm==key
        non_segm[idx] = SCANNET_COLOR_MAP_200[key]

    im_out = Image.fromarray(non_segm)
    im_out.show()

    


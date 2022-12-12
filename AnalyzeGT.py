from PIL import Image
import numpy as np
import glob
import os
import configs.scannet_constants

if __name__=="__main__":
    gt_dir = "/home/rozenberszki/Downloads/ScanNet-gt-png/"
    list_dirs = sorted(glob.glob(gt_dir+'/**', recursive=False))
    for scene in list_dirs:
        for image in sorted(glob.glob(os.path.join(gt_dir, scene, '**'), recursive=False)):
            print(image)
            exit()
    img = Image.open("/home/rozenberszki/Downloads/ScanNet-gt-png/plenoxel_scannet_scene0000_00/0.png")
    img = np.array(img)
    print(img)

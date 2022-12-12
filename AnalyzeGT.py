from PIL import Image
import numpy as np
import glob
import os
from configs.scannet_constants import *
from tqdm import tqdm
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--generate', action='store_true')
    args = parser.parse_args()

    save_needed = args.generate
    if(save_needed):
        gt_dir = "/home/rozenberszki/Downloads/ScanNet-gt-png/"
        list_dirs = sorted(glob.glob(gt_dir+'/**', recursive=False))
        class_dist = {}
        class_dist[0] = 0
        # for i in range(21):
        #     if(i==0): continue
        #     class_dist[VALID_CLASS_IDS_20[i-1]]=0
        for key in VALID_CLASS_IDS_20:
            class_dist[key]=0
        

        to_save_img = np.zeros((480,640,3), dtype=np.uint8)
        for tqdm_idx in tqdm(range(len(list_dirs))):
            scene = list_dirs[tqdm_idx]
            for image in sorted(glob.glob(os.path.join(gt_dir, scene, '**'), recursive=False)):
                img = Image.open(image)
                img = np.array(img)
                for key in range(21):
                    if(key==0): class_dist[key]+=(img==key).sum();continue
                    new_key = VALID_CLASS_IDS_20[key-1] ## Key
                    idx = img==key
                    class_dist[new_key]+= (img==key).sum()
                    # to_save_img[idx] = SCANNET_COLOR_MAP_20[new_key]

        # pil_img = Image.fromarray(to_save_img)
        # pil_img.save("./test.png")
        print(class_dist)

        np.save("../class_dist.npy", class_dist)

    load = np.load("../class_dist.npy", allow_pickle='TRUE').item()

    sum = 0
    for key in load.keys():
        sum+=load[key]

    print("Sum", sum)

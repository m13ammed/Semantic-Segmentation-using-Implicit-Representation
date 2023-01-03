from PIL import Image
import numpy as np
import glob
import os
from configs.scannet_constants import *
from tqdm import tqdm
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--generate', action='store_false')
    args = parser.parse_args()

    save_needed = args.generate
    if(save_needed):
        gt_dir = "/home/rozenberszki/Downloads/ScanNet-gt-124/"
        list_dirs = sorted(glob.glob(gt_dir+'/**', recursive=False))
        class_dist = {}
        class_dist[0] = {
            "sum":0,
            "scenes":{

            }
        }
        # for i in range(21):
        #     if(i==0): continue
        #     class_dist[VALID_CLASS_IDS_20[i-1]]=0
        for key in VALID_CLASS_IDS_20:
            class_dist[key]={
                "sum":0,
                "scenes":{

                }
            }

        scene_def = {
            "poses":[

            ]
        }
        

        for tqdm_idx in tqdm(range(len(list_dirs))):
            scene = list_dirs[tqdm_idx]
            for image in sorted(glob.glob(os.path.join(gt_dir, scene, '**'), recursive=False)):
                if(image.endswith(".png")): continue
                img = np.load(image, allow_pickle=True)
                scene_id = scene.split("/")[-1]
                pose_id = image.split("/")[-1]
                for key in range(21):
                    if(key==0): 
                        sum_pts = (img==key).sum()
                        class_dist[key]["sum"]+=(img==key).sum()
                        if(sum_pts > 0):
                            if(scene_id not in class_dist[key]["scenes"]):
                                class_dist[key]["scenes"][scene_id]={
                                "sum":(img==key).sum(),
                                "poses":{
                                    pose_id: (img==key).sum()
                                }  
                                }
                            else:
                                class_dist[key]["scenes"][scene_id]["sum"]+=(img==key).sum()
                                class_dist[key]["scenes"][scene_id]["poses"][pose_id] = (img==key).sum()
                        continue
                    new_key = VALID_CLASS_IDS_20[key-1] ## Key
                    # idx = img==key
                    sum_pts = (img==key).sum()
                    class_dist[new_key]["sum"]+= sum_pts
                    if(sum_pts > 0):
                        if(scene_id not in class_dist[new_key]["scenes"]):
                            class_dist[new_key]["scenes"][scene_id]={
                                "sum":(img==key).sum(),
                                "poses":{
                                pose_id: (img==key).sum()
                                }  
                            }
                        else:
                            class_dist[new_key]["scenes"][scene_id]["sum"]+=(img==key).sum()
                            class_dist[new_key]["scenes"][scene_id]["poses"][pose_id] = (img==key).sum()
                    # to_save_img[idx] = SCANNET_COLOR_MAP_20[new_key]

        # pil_img = Image.fromarray(to_save_img)
        # pil_img.save("./test.png")

        np.save("../class_dist_advanced.npy", class_dist)

    # # load = np.load("../class_dist_advanced.npy", allow_pickle='TRUE').item()

    # sum = 0
    # for key in load.keys():
    #     sum+=load[key]

    # print("Sum", sum)

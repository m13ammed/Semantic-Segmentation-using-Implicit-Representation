import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


from AD_project.configs.scannet_constants import SCANNET_COLOR_MAP_200
from utils.locations_constants import *
from PIL import Image
import numpy as np
import getopt


def visualize_gt(scene_name:str="scene0000_00", pose_id: int =0):
    gt_folder = "label-filt"
    scannet_img_path = os.path.join(original_scannet_folder, scene_name,gt_folder, str(pose_id)+".png")
    im_scannet = Image.open(scannet_img_path).resize((640,480))
    gt_segm = np.array(im_scannet)
    img_out = np.zeros((480,640,3), dtype=np.uint8)
    for key in SCANNET_COLOR_MAP_200.keys():
        idx = gt_segm==key
        img_out[idx] = SCANNET_COLOR_MAP_200[key]

    im_out = Image.fromarray(img_out)
    im_out.show()



if __name__ == "__main__":
    argv = sys.argv[1:]
    scene_name = None
    pose_id = None
    try:
      opts, args = getopt.getopt(argv,"hp:s:",["pose_id=","scene="])
    except getopt.GetoptError:
        print ('visualize_gt_png.py -s <scene_name> -p <pose_id>')
        sys.exit(2)

    for opt, arg in opts:
      if opt == '-h':
         print('visualize_gt_png.py -s <scene_name> -p <pose_id>')
         sys.exit()
      elif opt in ("-s", "--scene"):
         scene_name = arg
      elif opt in ("-p", "--pose_id"):
         pose_id = arg

    if((scene_name is not None) and (pose_id is not None)):
        visualize_gt(scene_name=scene_name, pose_id=pose_id)
    elif(scene_name is not None):
        visualize_gt(scene_name=scene_name)
    elif(pose_id is not None):
        visualize_gt(pose_id=pose_id)
    else:
        visualize_gt()
   
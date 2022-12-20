import os, sys
import argparse


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
from utils.export_images import export_images
#from utils.scannet_scene import ScanNetScene
from dataloader.litdata import LitDataPefceptionScannet
from dataloader.generate_groundtruth import generate_groundtruth_render
from pytorch3d.io import IO
from utils.ply_load import load_mesh_labels
import sys, getopt
import gin
import glob

@gin.configurable()
def render_seg_images(scannet_scene, compressed, show_only,frame_skip, datadir, scannet_dir, batch_size, colored = True):
    scannet_scene_name = scannet_scene
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0
    print(f"Generating Renders for scene {scannet_scene} {scannet_dir}")
    scannet_scene = LitDataPefceptionScannet(scene_name=scannet_scene, frame_skip=frame_skip, datadir=datadir, scannet_dir=scannet_dir, num_tpus=0, accelerator=device, num_gpus=num_gpus)
    segmented_mesh,labels, rgb = load_mesh_labels(scannet_scene.polygon, device)
    n_batches = len(scannet_scene.trans_info['frame_ids'])//batch_size
    if(len(scannet_scene.trans_info['frame_ids'])%batch_size!=0): n_batches+=1
    for batch in range(n_batches):
        labels, target_images = generate_groundtruth_render(
            scannet_scene=scannet_scene,
            mesh=segmented_mesh, 
            labels=labels,
            device=device, 
            batch_id=batch,
            batch_size=batch_size,
            compressed=compressed,
            rgb = rgb
        )
        export_images(labels=labels,target_images=target_images, show_only = show_only, batch=batch, batch_size=batch_size,frame_skip=frame_skip, scene_name=scannet_scene_name, colored=colored)

@gin.configurable()
def generate_all_scenes(compressed, show_only,frame_skip, datadir, scannet_dir, colored=True):
    list_dirs = sorted(glob.glob(scannet_dir+'/**', recursive=False))
    # start_idx = list_dirs.index(scannet_dir+"/scene0540_02")
    # list_dirs = list_dirs[start_idx:]
    for dir in list_dirs:
        scene_path = dir
        scene_name = "plenoxel_scannet_"+dir.split("/")[-1]
        render_seg_images(scannet_scene=scene_name, compressed=compressed, show_only=show_only,frame_skip=frame_skip, colored=col)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default="plenoxel_scannet_scene0001_00",
        help="scene name",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )

    parser.add_argument(
        "--compressed",
        type=bool,
        default=True,
        help="Export in 124x124",
    )

    parser.add_argument(
        "--show_only",
        type=bool,
        default=False,
        help="Show image",
    )

    parser.add_argument(
        "--colored",
        type=bool,
        default=False,
        help="Colored or grayscale",
    )

    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Number of frames to skip",
    )

    args = parser.parse_args()
    # argv = sys.argv[1:]
    compressed=args.compressed
    show_only=args.show_only
    ginc, ginb = ['/home/rozenberszki/karim/Semantic-Segmentation-using-Implicit-Representation/configs/render_scannet_seg_only.gin'], ['']
    scene_name = args.scene_name
    frame_skip = args.frame_skip
    col = args.colored
    # try:
    #     opts, args = getopt.getopt(argv,"hf:c:s:z:n",["frame_skip=","compressed=", "show_only=", "colored=","scene_name="])
    # except getopt.GetoptError:
    #     print ('main.py -f <frame_skip> -c <true|false> -s <true|false>')
    #     sys.exit(2)

    # for opt, arg in opts:
    #     if opt == '-h':
    #         print ('main.py -f <frame_skip> -c <true|false> -s <true|false> -n <scene_name_perf_format> -z <true|false>')
    #         sys.exit()
    #     elif opt in ("-f", "--frame_skip"):
    #         frame_skip = int(arg)
    #     elif opt in ("-c", "--compressed"):
    #         compressed = bool(arg)
    #     elif opt in ("-s", "--show_only"):
    #         show_only = bool(arg)
    #     elif opt in ("-ginc"):
    #         ginc.append(arg)
    #     elif opt in ("-ginb"):
    #         ginc.append(arg)
    #     elif opt in ("-n", "--scene_name"):
    #         scene_name = scene_name
    #         print("SCENE", scene_name)
    #     elif opt in ("-z", "--colored"):
    #         col = bool(arg)


    #ginbs = []
    gin.parse_config_files_and_bindings(ginc,ginb)
    
    # generate_all_scenes(compressed=compressed, show_only=show_only,frame_skip=frame_skip, colored=col)
    
    render_seg_images(scannet_scene=scene_name, compressed=compressed, show_only=show_only,frame_skip=frame_skip, colored=col)




    
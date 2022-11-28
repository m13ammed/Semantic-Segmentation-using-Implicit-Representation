import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import torch
from utils.dataloader import DataLoader
from utils.export_images import export_images
#from utils.scannet_scene import ScanNetScene
from dataloader.litdata import LitDataPefceptionScannet
from dataloader.generate_groundtruth import generate_groundtruth_render
from pytorch3d.io import IO
from utils.ply_load import load_mesh_labels
import sys, getopt
import gin

@gin.configurable()
def render_seg_images(scannet_scene, compressed, show_only,frame_skip, datadir, scannet_dir, batch_size):
    scannet_scene_name = scannet_scene
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0
    print(f"Generating Renders for scene {scannet_scene}")
    scannet_scene = LitDataPefceptionScannet(scene_name=scannet_scene, frame_skip=frame_skip, datadir=datadir, scannet_dir=scannet_dir, num_tpus=0, accelerator=device, num_gpus=num_gpus)
    segmented_mesh,labels, rgb = load_mesh_labels(scannet_scene.polygon, device)
    for batch in range(len(scannet_scene.trans_info['frame_ids'])):
        cameras, target_images = generate_groundtruth_render(
            scannet_scene=scannet_scene,
            mesh=segmented_mesh, 
            labels=labels,
            device=device, 
            batch_id=batch,
            batch_size=batch_size,
            compressed=compressed,
            rgb = rgb
        )
        export_images(target_images=target_images, show_only = show_only, batch=batch, batch_size=batch_size,frame_skip=frame_skip, scene_name=scannet_scene_name)


if __name__ == "__main__":
    argv = sys.argv[1:]
    compressed=False
    show_only=False
    ginc, ginb = ['../configs/render_scannet_seg_only.gin'], []
    scene_name = 'plenoxel_scannet_scene0000_00'
    frame_skip = 20
    try:
        opts, args = getopt.getopt(argv,"hf:c:s:",["frame_skip=","compressed=", "show_only="])
    except getopt.GetoptError:
        print ('main.py -f <frame_skip> -c <true|false> -s <true|false>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -f <frame_skip> -c <true|false> -s <true|false> -n <scene_name_perf_format>')
            sys.exit()
        elif opt in ("-f", "--frame_skip"):
            frame_skip = int(arg)
        elif opt in ("-c", "--compressed"):
            compressed = bool(arg)
        elif opt in ("-s", "--show_only"):
            show_only = bool(arg)
        elif opt in ("-ginc"):
            ginc.append(arg)
        elif opt in ("-ginb"):
            ginc.append(arg)
        elif opt in ("-n", "--scene_name"):
            scene_name = scene_name



    ginbs = []
    gin.parse_config_files_and_bindings(ginc,ginb)


    
    render_seg_images(scannet_scene=scene_name, compressed=compressed, show_only=show_only,frame_skip=frame_skip)




    
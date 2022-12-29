import os, sys

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
import argparse
@gin.configurable()
def render_seg_images(scannet_scene, compressed, show_only,frame_skip, datadir, scannet_dir, batch_size, colored):
    scannet_scene_name = scannet_scene
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0
    print(f"Generating Renders for scene {scannet_scene}")
    scannet_scene = LitDataPefceptionScannet(scene_name=scannet_scene, frame_skip=frame_skip, datadir=datadir, scannet_dir=scannet_dir, num_tpus=0, accelerator=device, num_gpus=num_gpus)
    segmented_mesh,labels, rgb = load_mesh_labels(scannet_scene.polygon, device)
    n_batches = len(scannet_scene.trans_info['frame_ids'])//batch_size
    if(len(scannet_scene.trans_info['frame_ids'])%batch_size!=0): n_batches+=1

    for batch in range(n_batches):
        labels_, target_images = generate_groundtruth_render(
            scannet_scene=scannet_scene,
            mesh=segmented_mesh, 
            labels=labels,
            device=device, 
            batch_id=batch,
            batch_size=batch_size,
            compressed=compressed,
            rgb = rgb
        )
        start_idx = batch*batch_size
        end_idx = start_idx + batch_size
        if (batch+1) == n_batches:
            frames = scannet_scene.trans_info['frame_ids'][start_idx:]
        else:
            frames = scannet_scene.trans_info['frame_ids'][start_idx:end_idx]
        export_images(labels=labels_,target_images=target_images, show_only = show_only, scene_name=scannet_scene_name, frame_ids = frames, colored = colored)


if __name__ == "__main__":
    argv = sys.argv[1:]
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
        default=False,
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
    ginc, ginb = ['/home/rozenberszki/mohammad/Semantic-Segmentation-using-Implicit-Representation/configs/render_scannet_seg_only.gin'], ['']
    scene_name = args.scene_name
    frame_skip = args.frame_skip
    col = args.colored



    #ginbs = []
    gin.parse_config_files_and_bindings(ginc,ginb)


    
    render_seg_images(scannet_scene=scene_name, compressed=compressed, show_only=show_only,frame_skip=frame_skip, colored = col)




    
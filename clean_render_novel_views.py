import os, sys
import argparse
from tqdm import tqdm


# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(ROOT_DIR)

import torch
# from utils.export_images import export_images
#from utils.scannet_scene import ScanNetScene
from dataloader.litdata import LitDataPefceptionScannet
from pytorch3d.io import IO
from utils.ply_load import load_mesh_labels
import sys, getopt
import gin
import argparse

from dataloader.litdata import LitDataPefceptionScannet
from utils.locations_constants import img_params
import torch
import numpy as np
from utils.SegmentationShader import SegmentationShader
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)

from PIL import Image
@gin.configurable()
def export_images(labels, target_images, depth_data, show_only=False, scene_name="1", frame_ids = [], output_segmentation = './out/', colored =False):
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for frame_id, label, imgg, depth in zip(frame_ids, labels, target_images, depth_data):
        frame_id = frame_id.split(".txt")[0]
        depth_out = np.array(depth.cpu(), dtype=np.float16)
        depth_out = depth_out.squeeze()
        seg = np.array(label.cpu(), dtype=np.uint8)
        cls_id = frame_id.split("_")[0]
        if(np.sum(seg==int(cls_id))==0): continue
        # print("DOES THE CLASS EXIST? COMPARING ", cls_id, "Sum:", np.sum(seg==cls_id))
        if(colored):
            image = (clamp_and_detach(imgg[ ..., :3]))
            tensor = image*255
            tensor = np.array(tensor, dtype=np.uint8)
            if np.ndim(tensor)>3:
                assert tensor.shape[0] == 1
                tensor = tensor[0]
            img_col = Image.fromarray(tensor)
        
        image = np.array(label.cpu(), dtype=np.uint8)
        img = image#Image.fromarray(image)
        if(show_only):
            img.show()
        else:
            os.makedirs(output_segmentation, exist_ok=True)
            path_scene = os.path.join(output_segmentation, scene_name)
            os.makedirs(path_scene, exist_ok=True)
            if(colored):
                img_col.save(os.path.join(path_scene,str(frame_id)+".png"))
            np.save(os.path.join(path_scene,str(frame_id)+".npy"),img)
            np.save(os.path.join(path_scene,str(frame_id)+"_depth.npy"),depth_out)


def generate_groundtruth_render(
    scannet_scene: LitDataPefceptionScannet,
    mesh ,
    labels,
    device:torch.device = torch.device("cpu"),
    batch_id: int =0,
    batch_size: int = 5,
    compressed=False,
    rgb = None,
    poses_list = None
):
    image_out_size = scannet_scene.image_sizes[0].tolist()
    # print(image_out_size)#[480, 640]
    if(compressed): image_out_size = [124,124]
    start_idx = batch_id*batch_size
    end_idx = start_idx + batch_size
    if(end_idx>=len(poses_list)): end_idx = len(poses_list)
    poses = poses_list[start_idx:end_idx].copy()
    # poses = scannet_scene.extrinsics[start_idx:end_idx].copy()
    if(poses.shape[0]!= batch_size): batch_size = poses.shape[0]
    if poses.ndim != 3:
        poses = np.expand_dim(poses, 0)
    R= poses[:,:3,:3].transpose(0,2,1)
    R[:,[1,0]] *= (-1)
    T = poses[:,:3,3:]
    T = -R @ T
    T = T.transpose(0,2,1)
    T = np.squeeze(T)
    R = R.transpose(0,2,1)
    # print(scannet_scene.trans_info['frame_ids'][start_idx:end_idx], start_idx, end_idx)
    #intrinsics = torch.tensor(scannet_scene.intrinsics).expand(poses.shape[0], -1, -1)
    intrinsics = scannet_scene.intrinsic_orig[:batch_size].copy()
    
    if np.array(intrinsics[:R.shape[0]]).ndim !=3 or R.ndim !=3 or T.ndim !=2:
        T = np.expand_dims(T, 0)
    # print("Hello?", start_idx, end_idx, len(scannet_scene.intrinsic_orig), intrinsics.shape, R.shape, T.shape)
    cameras = PerspectiveCameras(
        # focal_length=((-cam_params['fx'], -cam_params['fy']),),
        # principal_point=((cam_params['mx'], cam_params['my']),),
        in_ndc=False,
        image_size=((img_params['height'],img_params['width']),),
        K=np.array(intrinsics[:R.shape[0]]),
        device=device,
        # K = [intrinsic],
        R=np.array(R),
        T=np.array(T)
    )
    ### Bug where apparently for 1 items it creates a minimum of 3 cameras. 
    if(len(poses_list[start_idx:end_idx])==1):
        cameras = cameras[0]

    raster_settings = RasterizationSettings(
        image_size=image_out_size, blur_radius=0.0, faces_per_pixel=1, bin_size=None
    ) 
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SegmentationShader(
            device=device, cameras=cameras
        ),
    )
    meshes = mesh.extend(poses.shape[0])

    # Render the  mesh from each viewing angle
    labels_, target_images, depth = renderer(meshes, cameras=cameras, labels=labels, rgb= rgb)
    meshes.to("cpu")
    labels.to("cpu")
    if rgb is not None:
        rgb.to("cpu")
    return labels_, target_images, depth#[..., :3]


@gin.configurable()
def render_novel_seg(scannet_scene, compressed, show_only,frame_skip, datadir, scannet_dir, batch_size, colored, pose_dir):
    scannet_scene_name = scannet_scene
    if(scannet_scene_name.endswith("_01") or scannet_scene_name.endswith("_02")): exit()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")
    num_gpus = 1 if torch.cuda.is_available() else 0
    print(f"Generating Renders for scene {scannet_scene} {scannet_dir}")
    scannet_scene = LitDataPefceptionScannet(scene_name=scannet_scene, frame_skip=frame_skip, datadir=datadir, scannet_dir=scannet_dir, num_tpus=0, accelerator=device, num_gpus=num_gpus)
    segmented_mesh,labels, rgb = load_mesh_labels(scannet_scene.polygon, device)
    ### Load novel poses
    scene_pose_dir = os.path.join(pose_dir, scannet_scene_name, "pose")
    poses_list = []
    for p in os.listdir(scene_pose_dir):
        p_path = os.path.join(scene_pose_dir, p)
        pose_np = np.loadtxt(p_path)
        poses_list.append(pose_np)
    poses_list = np.array(poses_list)
    n_batches = poses_list.shape[0]//batch_size
    if(len(poses_list)%batch_size!=0): n_batches+=1
    for batch in tqdm(range(n_batches)):
        labels_, target_images, depth = generate_groundtruth_render(
            scannet_scene=scannet_scene,
            mesh=segmented_mesh, 
            labels=labels,
            device=device, 
            batch_id=batch,
            batch_size=batch_size,
            compressed=compressed,
            rgb = rgb,
            poses_list = poses_list
        )
        start_idx = batch*batch_size
        end_idx = start_idx + batch_size
        pose_ids = os.listdir(scene_pose_dir)
        if (batch+1) == n_batches:
            frames = pose_ids[start_idx:]
        else:
            frames = pose_ids[start_idx:end_idx]

        export_images(labels=labels_,target_images=target_images, depth_data=depth, show_only = show_only, scene_name=scannet_scene_name, frame_ids = frames, colored = colored)


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser()    # argv = sys.argv[1:]
    parser.add_argument(
        "--compressed",
        type=bool,
        default=True,
        help="Export in 124x124",
    )

    parser.add_argument(
        "--scene_name",
        type=str,
        default=True,
        help="Export in 124x124",
    )


    parser.add_argument(
        "--colored",
        type=bool,
        default=False,
        help="Colored or grayscale",
    )

    parser.add_argument(
        "--show_only",
        type=bool,
        default=False,
        help="Show only?",
    )


    parser.add_argument(
        "--frame_skip",
        type=int,
        default=1,
        help="Do we skip frames?",
    )

    args = parser.parse_args()
    # argv = sys.argv[1:]
    compressed=args.compressed
    ginc, ginb = ['/home/rozenberszki/karim/Semantic-Segmentation-using-Implicit-Representation/configs/render_novel_seg.gin'], ['']
    scene_name = args.scene_name
    frame_skip = args.frame_skip
    col = args.colored
    show_only = args.show_only
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
    
    render_novel_seg(scannet_scene=scene_name, compressed=compressed, show_only=show_only,frame_skip=frame_skip, colored = col)




    
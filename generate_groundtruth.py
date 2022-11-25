from utils.scannet_scene import ScanNetScene
from utils.locations_constants import img_params
import torch
import numpy as np
from utils.SegmentationShader import SegmentationShader
from pytorch3d.renderer import (
    PointLights, 
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
)

def generate_groundtruth_render(
    scannet_scene: ScanNetScene,
    device:torch.device = torch.device("cpu"),
    batch_id: int =0,
    batch_size: int = 5,
    mesh = None,
    compressed=False,
):
    image_out_size = [480, 640]
    if(compressed): image_out_size = [128,128]
    start_idx = batch_id*batch_size
    end_idx = start_idx + batch_size
    if(end_idx>scannet_scene.poses.shape[0]): end_idx = scannet_scene.poses.shape[0]-1
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    poses = scannet_scene.poses[start_idx:end_idx].copy()
    R= poses[:,:3,:3].transpose(0,2,1)
    R[:,[1,0]] *= (-1)
    T = poses[:,:3,3:]
    T = -R @ T
    T = T.transpose(0,2,1)
    T = np.squeeze(T)
    R = R.transpose(0,2,1)
    print(scannet_scene.pose_ids[start_idx:end_idx])
    intrinsics = torch.tensor(scannet_scene.intrinsics).unsqueeze(0).expand(poses.shape[0], -1, -1)


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

    raster_settings = RasterizationSettings(
        image_size=image_out_size, blur_radius=0.0, faces_per_pixel=1
    ) 
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SegmentationShader(
            device=device, cameras=cameras
        ),
    )
    meshes = mesh.extend(poses.shape[0])

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    return cameras, target_images[..., :3]

    



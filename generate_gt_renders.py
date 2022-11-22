from pytorch3d.io import IO
import torch
import numpy as np
from locations_const import *
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    PerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)

from read_scene import Scene
def generate_gt_renders(
    poses, num_views: int = 1, polygon_path: str = "seg.ply", device = torch.device("cpu"), intrinsic = None
):
    mesh = IO().load_mesh(polygon_path, device=device)
    IO().save_mesh(mesh, "out.ply")
    
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    # mesh.offset_verts_(-(center.expand(N, 3)))
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    M = torch.eye(3).to(device=device)
    ## Flip Y and Z
    # M[0,0]=1
    # M[1,1]=0
    # M[1,2]=1
    # M[2,2]=0
    # M[2,1]=1

    ### Flip X and Y
    # M[0,0]=0
    # M[0,1]=1
    # M[1,1]=0
    # M[1,0]=1
    # M[2,2]=1
    # M[2,1]=0

    pose = poses[0]
    pose = torch.tensor(pose)
    pose[0,1]=-pose[0,1]
    pose = pose.inverse()
    # pose = torch.linalg.inv(pose).contiguous()


    # M[0,0]=-1.0
    # M[1,1]=-1.0
    # M[2,2]=-1.0

    R = pose[:3,:3].T
    # R[:,2] *= -1
    # R = R.T
    # print(R)
    # R[2,] *= -1
    # R = R.T
    print(R)
    # print(R)
    T = pose[:3,3:]
    # print(T)
    # print(T, T.shape)
    # R = M @ R @ M.inverse()
    R= np.array(R)
    # print(R)
    T = M @ T
    T = np.array(T)
 
    # cameras = FoVPerspectiveCameras(device=device,R=np.array([R]), T=T.T)
    cameras = PerspectiveCameras(
        focal_length=((cam_params['fy'], cam_params['fx']),),
        principal_point=((cam_params['my'], cam_params['mx']),),
        in_ndc=False,
        image_size=((img_params['height'], img_params['width']),),
        device=device,
        # K = [intrinsic],
        R=np.array([R]),
        T=T.T
    )
    raster_settings = RasterizationSettings(
        # image_size=[480, 640], blur_radius=0.0, faces_per_pixel=1
        image_size=128, blur_radius=0.0, faces_per_pixel=1
    ) 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras[0], lights=lights)
    return cameras, target_images[..., :3]



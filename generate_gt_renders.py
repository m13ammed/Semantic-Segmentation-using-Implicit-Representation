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
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

class SegmentationShader(ShaderBase):
    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        texels = meshes.sample_textures(fragments).clone()
        return texels[...,0,:]

from read_scene import Scene
def generate_gt_renders(
    poses, num_views: int = 1, polygon_path: str = "seg.ply", device = torch.device("cpu"), intrinsic = None
):
    mesh = IO().load_mesh(polygon_path, device=device)
    
    # verts = mesh.verts_packed()
    # N = verts.shape[0]
    # center = verts.mean(0)
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
    print(pose)
    # pose = pose.inverse()
    # pose = torch.linalg.inv(pose).contiguous()


    # M[0,0]=-1.0
    # M[1,1]=-1.0
    # M[2,2]=-1.0

    R = pose[:3,:3].T
    R[[1,0]] *= (-1)
    # R[1,2]*=-1
    # R = R.T
    # print(R)
    # R[2,] *= -1
    # R = R.T
    print(R)
    # print(R)
    # R *=-1
    T = pose[:3,3:]
    T = -R @ T
    # T[2] *=-1
    # print(T)
    # print(T, T.shape)
    # R = M @ R @ M.inverse()
    R= np.array(R)
    # print(R)
    # T = M @ T
    T = np.array(T)
 
    # cameras = FoVPerspectiveCameras(device=device,R=np.array([R]), T=T.T)
    cameras = PerspectiveCameras(
        # focal_length=((-cam_params['fx'], -cam_params['fy']),),
        # principal_point=((cam_params['mx'], cam_params['my']),),
        in_ndc=False,
        image_size=((img_params['height'],img_params['width']),),
        K=np.array([intrinsic]),
        device=device,
        # K = [intrinsic],
        R=np.array([R.T]),
        T=T.T
    )
    raster_settings = RasterizationSettings(
        image_size=[480, 640], blur_radius=0.0, faces_per_pixel=1
        # image_size=128, blur_radius=0.0, faces_per_pixel=1
    ) 
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SegmentationShader(
            device=device, cameras=cameras
        ),
    )
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras[0], lights=lights)
    return cameras, target_images[..., :3]



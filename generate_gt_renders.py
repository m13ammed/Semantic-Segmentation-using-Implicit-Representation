from pytorch3d.io import IO
import torch
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
)
def generate_gt_renders(
    poses, num_views: int = 1, polygon_path: str = "seg.ply", device = torch.device("cpu")
):
    mesh = IO().load_mesh(polygon_path, device=device)
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    mesh.offset_verts_(-(center.expand(N, 3)))
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])


    R = poses[0][:3,:3]
    T = poses[0][:3,3:]
    print(R,T)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    print(cameras)
    raster_settings = RasterizationSettings(
        image_size=[480, 640], blur_radius=0.0, faces_per_pixel=1
        # image_size=128, blur_radius=0.0, faces_per_pixel=1
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
    target_images = renderer(meshes, cameras=cameras, lights=lights)
    return cameras, target_images[..., :3]



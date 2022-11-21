# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
from pytorch3d.io import load_objs_as_meshes, IO
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    BlendParams,
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftSilhouetteShader,
)

from read_scene import Scene
from locations import *
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot





# create the default data directory
current_dir = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(current_dir, "..", "data", "cow_mesh")


def generate_cow_renders(
    num_views: int = 1, data_dir: str = DATA_DIR, azimuth_range: float = 180
):
    """
    This function generates `num_views` renders of a cow mesh.
    The renders are generated from viewpoints sampled at uniformly distributed
    azimuth intervals. The elevation is kept constant so that the camera's
    vertical position coincides with the equator.

    For a more detailed explanation of this code, please refer to the
    docs/tutorials/fit_textured_mesh.ipynb notebook.

    Args:
        num_views: The number of generated renders.
        data_dir: The folder that contains the cow mesh files. If the cow mesh
            files do not exist in the folder, this function will automatically
            download them.
        azimuth_range: number of degrees on each side of the start position to
            take samples

    Returns:
        cameras: A batch of `num_views` `FoVPerspectiveCameras` from which the
            images are rendered.
        images: A tensor of shape `(num_views, height, width, 3)` containing
            the rendered images.
        silhouettes: A tensor of shape `(num_views, height, width)` containing
            the rendered silhouettes.
    """
    
    # set the paths

    # download the cow mesh if not done before
    cow_mesh_files = [
        os.path.join(data_dir, fl) for fl in ("cow.obj", "cow.mtl", "cow_texture.png")
    ]
    if any(not os.path.isfile(f) for f in cow_mesh_files):
        os.makedirs(data_dir, exist_ok=True)
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.obj"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow.mtl"
        )
        os.system(
            f"wget -P {data_dir} "
            + "https://dl.fbaipublicfiles.com/pytorch3d/data/cow_mesh/cow_texture.png"
        )

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else: device = torch.device("cpu")


    # Load obj file
    obj_filename = os.path.join(data_dir, "cow.obj")
    # mesh = load_objs_as_meshes([obj_filename], device=device)
    mesh = IO().load_mesh(os.path.join(data_dir, "segm.ply"), device=device)
#     scene = Scene(
#     perf_folder= perfception_scannet_folder,
#     orig_scannet_folder= original_scannet_folder,
#     scene_name=scene_name
# )
#     pts = scene.links_idx.numpy().astype(np.float64)
#     pcd_o3d = o3d.geometry.PointCloud()
#     pcd_o3d.points = o3d.utility.Vector3dVector(pts)
#     mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd_o3d, 1)
#     mesh = Meshes(verts=mesh_o3d.vertices, faces=mesh_o3d.triangles)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh
    # to its original center and scale.  Note that normalizing the target mesh,
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-(center.expand(N, 3)))
    # mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles.
    elev = torch.linspace(90, 90, num_views)    # keep constant
    azim = torch.linspace(-azimuth_range, azimuth_range, num_views)+180

    ## Defaults
    # elev = torch.linspace(0,0, num_views)# keep constant
    # azim = torch.linspace(-azimuth_range, azimuth_range, num_views) + 180

    # Place a point light in front of the object. As mentioned above, the front of
    # the cow is facing the -z direction.
    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different
    # viewing angles. All the cameras helper methods support mixed type inputs and
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and
    # then specify elevation and azimuth angles for each viewpoint as tensors.
    R, T = look_at_view_transform(dist=1, elev=elev, azim=azim)
    rx = Rot.from_euler('x', 90, degrees=True)
    r1 = (Rot.from_euler('z', 0, degrees=True)*rx).as_matrix()
    r2 = (Rot.from_euler('z', 90, degrees=True)*rx).as_matrix()
    r3 = (Rot.from_euler('z', 180, degrees=True)*rx).as_matrix()
    r4 = (Rot.from_euler('z', 270, degrees=True)*rx).as_matrix()

    cameras = FoVPerspectiveCameras(device=device, R=np.array([r1,r2,r3,r4]), T=T)

    # Define the settings for rasterization and shading. Here we set the output
    # image to be of size 128X128. As we are rendering images for visualization
    # purposes only we will set faces_per_pixel=1 and blur_radius=0.0. Refer to
    # rasterize_meshes.py for explanations of these parameters.  We also leave
    # bin_size and max_faces_per_bin to their default values of None, which sets
    # their values using heuristics and ensures that the faster coarse-to-fine
    # rasterization method is used.  Refer to docs/notes/renderer.md for an
    # explanation of the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=[480, 640], blur_radius=0.0, faces_per_pixel=1
        # image_size=128, blur_radius=0.0, faces_per_pixel=1
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured
    # Phong shader will interpolate the texture uv coordinates for each vertex,
    # sample from a texture image and apply the Phong lighting model
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0.0, 0.0, 0.0))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device, cameras=cameras, lights=lights, blend_params=blend_params
        ),
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures.
    # Meshes has a useful `extend` method which allows us do this very easily.
    # This also extends the textures.
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras, lights=lights)

    # # Rasterization settings for silhouette rendering
    # sigma = 1e-4
    # raster_settings_silhouette = RasterizationSettings(
    #     image_size=128, blur_radius=np.log(1.0 / 1e-4 - 1.0) * sigma, faces_per_pixel=50
    # )

    # # Silhouette renderer
    # renderer_silhouette = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, raster_settings=raster_settings_silhouette
    #     ),
    #     shader=SoftSilhouetteShader(),
    # )

    # # Render silhouette images.  The 3rd channel of the rendering output is
    # # the alpha/silhouette channel
    # silhouette_images = renderer_silhouette(meshes, cameras=cameras, lights=lights)

    # # binary silhouettes
    # silhouette_binary = (silhouette_images[..., 3] > 1e-4).float()

    return cameras, target_images[..., :3]
    return cameras, target_images[..., :3], silhouette_binary


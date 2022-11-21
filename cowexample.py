import os
import sys
import time
import json
import glob
import torch
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from IPython import display
import torchvision.transforms as T


# Data structures and functions for rendering
from pytorch3d.structures import Volumes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, 
    VolumeRenderer,
    NDCMultinomialRaysampler,
    EmissionAbsorptionRaymarcher
)
from pytorch3d.transforms import so3_exp_map
from read_scene import Scene

# obtain the utilized device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else: device = torch.device("cpu")


from utils.generate_cow_renders import generate_cow_renders
from utils.plot_image_grid import image_grid

def generate_rotating_volume(volume_model, n_frames = 50):
    logRs = torch.zeros(n_frames, 3, device=device)
    logRs[:, 1] = torch.linspace(0.0, 2.0 * 3.14, n_frames, device=device)
    Rs = so3_exp_map(logRs)
    Ts = torch.zeros(n_frames, 3, device=device)
    Ts[:, 2] = 2.7
    frames = []
    print('Generating rotating volume ...')
    for R, T in zip(tqdm(Rs), Ts):
        camera = FoVPerspectiveCameras(
            R=R[None], 
            T=T[None], 
            znear = target_cameras.znear[0],
            zfar = target_cameras.zfar[0],
            aspect_ratio = target_cameras.aspect_ratio[0],
            fov = target_cameras.fov[0],
            device=device,
        )
        frames.append(volume_model(camera)[..., :3].clamp(0.0, 1.0))
    return torch.cat(frames)
    


class VolumeModel(torch.nn.Module):
    def __init__(self, renderer, volume_size=[64] * 3, voxel_size=0.1):
        super().__init__()
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # densities close to zero.
        self.log_densities = torch.nn.Parameter(-4.0 * torch.ones(1, *volume_size))
        # After evaluating torch.sigmoid(self.log_colors), we get 
        # a neutral gray color everywhere.
        self.log_colors = torch.nn.Parameter(torch.zeros(3, *volume_size))
        self._voxel_size = voxel_size
        # Store the renderer module as well.
        self._renderer = renderer
        
    def forward(self, cameras):
        batch_size = cameras.R.shape[0]

        # Convert the log-space values to the densities/colors
        densities = torch.sigmoid(self.log_densities)
        colors = torch.sigmoid(self.log_colors)
        
        # Instantiate the Volumes object, making sure
        # the densities and colors are correctly
        # expanded batch_size-times.
        volumes = Volumes(
            densities = densities[None].expand(
                batch_size, *self.log_densities.shape),
            features = colors[None].expand(
                batch_size, *self.log_colors.shape),
            voxel_size=self._voxel_size,
        )
        
        # Given cameras and volumes, run the renderer
        # and return only the first output value 
        # (the 2nd output is a representation of the sampled
        # rays which can be omitted for our purpose).
        return self._renderer(cameras=cameras, volumes=volumes)[0]
    
# A helper function for evaluating the smooth L1 (huber) loss
# between the rendered silhouettes and colors.
def huber(x, y, scaling=0.1):
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    return loss




if __name__ == "__main__":
    perfception_scannet_folder = "../PeRFception-ScanNet"
    original_scannet_folder = "../ScanNet/scans"
    scene_name = "plenoxel_scannet_scene0000_00"
    img_dir = "render_model"
    img_name = "image000.jpg"
    render_dir = "./images"
    cam_params = {
        "fx":1170.187988,
        "fy":1170.187988,
        "mx":647.750000,
        "my":483.750000
    } # Color
    img_params = {
        "width":1296,
        "height":968
    } ## Probabbly high res. Low res might be 640, 480
    scene = Scene(
        perf_folder= perfception_scannet_folder,
        orig_scannet_folder= original_scannet_folder,
        scene_name=scene_name
    )
    # target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=4)
    target_cameras, target_images = generate_cow_renders(num_views=4)
    clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
    for idx in range(target_images.shape[0]):
        image = (clamp_and_detach(target_images[idx, ..., :3]))
        tensor = image*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        img = Image.fromarray(tensor)
        img.save(os.path.join("./out/",str(idx)+".png"))
    exit()
    for image in target_images[0]:
        transform = T.ToPILImage()
        print(image.shape)

    print(f'Generated {len(target_images)} images/silhouettes/cameras.')

    # render_size describes the size of both sides of the 
    # rendered images in pixels. We set this to the same size
    # as the target images. I.e. we render at the same
    # size as the ground truth images.
    render_size = target_images.shape[1]

    # Our rendered scene is centered around (0,0,0) 
    # and is enclosed inside a bounding box
    # whose side is roughly equal to 3.0 (world units).
    volume_extent_world = 3.0

    # 1) Instantiate the raysampler.
    # Here, NDCMultinomialRaysampler generates a rectangular image
    # grid of rays whose coordinates follow the PyTorch3D
    # coordinate conventions.
    # Since we use a volume of size 128^3, we sample n_pts_per_ray=150,
    # which roughly corresponds to a one ray-point per voxel.
    # We further set the min_depth=0.1 since there is no surface within
    # 0.1 units of any camera plane.
    raysampler = NDCMultinomialRaysampler(
        image_width=render_size,
        image_height=render_size,
        n_pts_per_ray=150,
        min_depth=0.1,
        max_depth=volume_extent_world,
    )


    # 2) Instantiate the raymarcher.
    # Here, we use the standard EmissionAbsorptionRaymarcher 
    # which marches along each ray in order to render
    # each ray into a single 3D color vector 
    # and an opacity scalar.
    raymarcher = EmissionAbsorptionRaymarcher()

    # Finally, instantiate the volumetric render
    # with the raysampler and raymarcher objects.
    renderer = VolumeRenderer(
        raysampler=raysampler, raymarcher=raymarcher,
    )

    # First move all relevant variables to the correct device.
    target_cameras = target_cameras.to(device)
    target_images = target_images.to(device)
    target_silhouettes = target_silhouettes.to(device)

    # Instantiate the volumetric model.
    # We use a cubical volume with the size of 
    # one side = 128. The size of each voxel of the volume 
    # is set to volume_extent_world / volume_size s.t. the
    # volume represents the space enclosed in a 3D bounding box
    # centered at (0, 0, 0) with the size of each side equal to 3.
    volume_size = 128
    volume_model = VolumeModel(
        renderer,
        volume_size=[volume_size] * 3, 
        voxel_size = volume_extent_world / volume_size,
    ).to(device)

    # Instantiate the Adam optimizer. We set its master learning rate to 0.1.
    lr = 0.1
    optimizer = torch.optim.Adam(volume_model.parameters(), lr=lr)

    # We do 300 Adam iterations and sample 10 random images in each minibatch.
    batch_size = 10
    n_iter = 300
    for iteration in range(n_iter):

        # In case we reached the last 75% of iterations,
        # decrease the learning rate of the optimizer 10-fold.
        if iteration == round(n_iter * 0.75):
            print('Decreasing LR 10-fold ...')
            optimizer = torch.optim.Adam(
                volume_model.parameters(), lr=lr * 0.1
            )
        
        # Zero the optimizer gradient.
        optimizer.zero_grad()
        
        # Sample random batch indices.
        batch_idx = torch.randperm(len(target_cameras))[:batch_size]
        
        # Sample the minibatch of cameras.
        batch_cameras = FoVPerspectiveCameras(
            R = target_cameras.R[batch_idx], 
            T = target_cameras.T[batch_idx], 
            znear = target_cameras.znear[batch_idx],
            zfar = target_cameras.zfar[batch_idx],
            aspect_ratio = target_cameras.aspect_ratio[batch_idx],
            fov = target_cameras.fov[batch_idx],
            device = device,
        )
        
        # Evaluate the volumetric model.
        rendered_images, rendered_silhouettes = volume_model(
            batch_cameras
        ).split([3, 1], dim=-1)
        
        # Compute the silhouette error as the mean huber
        # loss between the predicted masks and the
        # target silhouettes.
        sil_err = huber(
            rendered_silhouettes[..., 0], target_silhouettes[batch_idx],
        ).abs().mean()

        # Compute the color error as the mean huber
        # loss between the rendered colors and the
        # target ground truth images.
        color_err = huber(
            rendered_images, target_images[batch_idx],
        ).abs().mean()
        
        # The optimization loss is a simple
        # sum of the color and silhouette errors.
        loss = color_err + sil_err 
        
        # Print the current values of the losses.
        if iteration % 10 == 0:
            print(
                f'Iteration {iteration:05d}:'
                + f' color_err = {float(color_err):1.2e}'
                + f' mask_err = {float(sil_err):1.2e}'
            )
        
        # Take the optimization step.
        loss.backward()
        optimizer.step()
        
        # Visualize the renders every 40 iterations.
        if iteration % 40 == 0:
            # Visualize only a single randomly selected element of the batch.
            im_show_idx = int(torch.randint(low=0, high=batch_size, size=(1,)))
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            ax = ax.ravel()
            clamp_and_detach = lambda x: x.clamp(0.0, 1.0).cpu().detach().numpy()
            ax[0].imshow(clamp_and_detach(rendered_images[im_show_idx]))
            ax[1].imshow(clamp_and_detach(target_images[batch_idx[im_show_idx], ..., :3]))
            ax[2].imshow(clamp_and_detach(rendered_silhouettes[im_show_idx, ..., 0]))
            ax[3].imshow(clamp_and_detach(target_silhouettes[batch_idx[im_show_idx]]))
            for ax_, title_ in zip(
                ax, 
                ("rendered image", "target image", "rendered silhouette", "target silhouette")
            ):
                ax_.grid("off")
                ax_.axis("off")
                ax_.set_title(title_)
            fig.canvas.draw(); fig.show()
            display.clear_output(wait=True)
            display.display(fig)

    with torch.no_grad():
        rotating_volume_frames = generate_rotating_volume(volume_model, n_frames=7*4)

    image_grid(rotating_volume_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=7, rgb=True, fill=True)
    plt.show()

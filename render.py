import os
import json
import numpy as np
import open3d as o3d
import torch
from read_scene import Scene

print_separator = "==========="
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

def normalize(arr):
    return (arr-arr.min())/(arr.max()-arr.min())


def vis_vox(links_idx, density):
    """
    Visualizes the voxels using O3D
    Uses density as colors
    """
    pts = links_idx.numpy().astype(np.float64)
    pts_color = (density - density.min()) / (density.max() - density.min())
    pts_color = pts_color.numpy().astype(np.float64).repeat(3, axis=-1)
    pts = np.concatenate([pts], axis=0)     
    pts_color = np.concatenate([pts_color], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(pts_color)
    o3d.visualization.draw_geometries([pcd])


def generate_rays(cam_params, img_params): 
    """
    Generate rays, plenoxels svox2.py (gen_rays)
    NOTE: Look up NDC usage
    :param cam_paramms: dict
    :param img_params: dict
    :return: (origins (H*W, 3), dirs (H*W, 3))
    """
    height = img_params['height']
    width = img_params['width']
    fx = cam_params['fx']
    fy = cam_params['fy']
    mx = cam_params['mx']
    my = cam_params['my']


    # origins = c2w[None, :3, 3].expand(height*width, -1).contigous()
    origins = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    origins = torch.tensor(origins)
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float64) + 0.5,
        torch.arange(width, dtype=torch.float64) + 0.5,
    )
    xx = (xx - mx) / fx
    yy = (yy - my) / fy
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)   # OpenCV
    del xx, yy, zz
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3, 1)
    # dirs = (c2w[None, :3, :3].double() @ dirs)[..., 0]
    dirs = dirs.reshape(-1, 3).float()
    return {'origins':origins, 'dirs':dirs}
    
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);
            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def _fetch_links(links, scene: Scene):
    sh_data = scene.sh_data
    density_data = scene.density
    results_sigma = torch.zeros(
        (links.size(0), 1), device=links.device, dtype=torch.float32
    )
    results_sh = torch.zeros(
        (links.size(0), sh_data.size(1)),
        device=links.device,
        dtype=torch.float32,
    )
    mask = links >= 0
    idxs = links[mask].long()
    results_sigma[mask] = density_data[idxs]
    results_sh[mask] = sh_data[idxs]
    return results_sigma, results_sh

def render_img(scene: Scene, cam_params, img_params, batch_size: int = 1):
    grid = scene.links_idx
    rays = generate_rays(cam_params=cam_params, img_params=img_params)
    all_rgb_out = []
    for batch_start in range(0, img_params['height']*img_params['width'], batch_size):
        # per pixel/ray operation
        ## missing world2cam waiting till understand if it is needed.
        dirs = rays['dirs'] / torch.norm(rays['dirs'], dim=-1, keepdim=True)
        viewdirs = dirs
        B = dirs.size(0)
        origins = rays['origins']
        origins = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
        # assert rays['origins'].size(0) == B
        delta_scale = 1.0 / dirs.norm(dim=1)
        dirs *= delta_scale.unsqueeze(-1)
        sh_mult = eval_sh_bases(9, viewdirs)
        invdirs = 1.0 / dirs
        gsz = torch.tensor(grid.shape, device="cpu", dtype=torch.float32)
        gsz_cu = gsz.to(device=dirs.device)
        t1 = (-0.5 - origins) * invdirs
        t2 = (gsz_cu - 0.5 - origins) * invdirs

        t = torch.min(t1, t2)
        t[dirs == 0] = -1e9
        t = torch.max(t, dim=-1).values.clamp_min_(0)

        tmax = torch.max(t1, t2)
        tmax[dirs == 0] = 1e9
        tmax = torch.min(tmax, dim=-1).values

        log_light_intensity = torch.zeros(B, device=origins.device)
        out_rgb = torch.zeros((B, 3), device=origins.device)
        good_indices = torch.arange(B, device=origins.device)

        origins_ini = origins
        dirs_ini = dirs

        mask = t <= tmax
        good_indices = good_indices[mask]
        origins = origins[mask]
        dirs = dirs[mask]

        #  invdirs = invdirs[mask]
        del invdirs
        t = t[mask]
        sh_mult = sh_mult[mask]
        tmax = tmax[mask]


        while good_indices.numel() > 0:
            pos = origins + t[:, None] * dirs
            pos = pos.clamp_min_(0.0)
            pos[:, 0] = torch.clamp_max(pos[:, 0], gsz[0] - 1)
            pos[:, 1] = torch.clamp_max(pos[:, 1], gsz[1] - 1)
            pos[:, 2] = torch.clamp_max(pos[:, 2], gsz[2] - 1)
            #  print('pym', pos, log_light_intensity)

            l = pos.to(torch.long)
            l.clamp_min_(0)
            l[:, 0] = torch.clamp_max(l[:, 0], gsz[0] - 2)
            l[:, 1] = torch.clamp_max(l[:, 1], gsz[1] - 2)
            l[:, 2] = torch.clamp_max(l[:, 2], gsz[2] - 2)
            pos -= l

            # BEGIN CRAZY TRILERP
            lx, ly, lz = l.unbind(-1)
            links000 = grid[lx, ly, lz]
            links001 = grid[lx, ly, lz + 1]
            links010 = grid[lx, ly + 1, lz]
            links011 = grid[lx, ly + 1, lz + 1]
            links100 = grid[lx + 1, ly, lz]
            links101 = grid[lx + 1, ly, lz + 1]
            links110 = grid[lx + 1, ly + 1, lz]
            links111 = grid[lx + 1, ly + 1, lz + 1]

            sigma000, rgb000 = _fetch_links(links000, scene=scene)
            sigma001, rgb001 = _fetch_links(links001, scene=scene)
            sigma011, rgb011 = _fetch_links(links011, scene=scene)
            sigma100, rgb100 = _fetch_links(links100, scene=scene)
            sigma101, rgb101 = _fetch_links(links101, scene=scene)
            sigma010, rgb010 = _fetch_links(links010, scene=scene)
            sigma110, rgb110 = _fetch_links(links110, scene=scene)
            sigma111, rgb111 = _fetch_links(links111, scene=scene)

            wa, wb = 1.0 - pos, pos
            c00 = sigma000 * wa[:, 2:] + sigma001 * wb[:, 2:]
            c01 = sigma010 * wa[:, 2:] + sigma011 * wb[:, 2:]
            c10 = sigma100 * wa[:, 2:] + sigma101 * wb[:, 2:]
            c11 = sigma110 * wa[:, 2:] + sigma111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            sigma = c0 * wa[:, :1] + c1 * wb[:, :1]

            c00 = rgb000 * wa[:, 2:] + rgb001 * wb[:, 2:]
            c01 = rgb010 * wa[:, 2:] + rgb011 * wb[:, 2:]
            c10 = rgb100 * wa[:, 2:] + rgb101 * wb[:, 2:]
            c11 = rgb110 * wa[:, 2:] + rgb111 * wb[:, 2:]
            c0 = c00 * wa[:, 1:2] + c01 * wb[:, 1:2]
            c1 = c10 * wa[:, 1:2] + c11 * wb[:, 1:2]
            rgb = c0 * wa[:, :1] + c1 * wb[:, :1]

            # END CRAZY TRILERP

            log_att = (
                -0.5
                * torch.relu(sigma[..., 0])
                * delta_scale[good_indices]
            )
            weight = torch.exp(log_light_intensity[good_indices]) * (
                1.0 - torch.exp(log_att)
            )
            # [B', 3, n_sh_coeffs]
            rgb_sh = rgb.reshape(-1, 3, 9)
            rgb = torch.clamp_min(
                torch.sum(sh_mult.unsqueeze(-2) * rgb_sh, dim=-1) + 0.5,
                0.0,
            )  # [B', 3]
            rgb = weight[:, None] * rgb[:, :3]

            out_rgb[good_indices] += rgb
            log_light_intensity[good_indices] += log_att
            t += 0.5

            mask = t <= tmax
            good_indices = good_indices[mask]
            origins = origins[mask]
            dirs = dirs[mask]
            #  invdirs = invdirs[mask]
            t = t[mask]
            sh_mult = sh_mult[mask]
            tmax = tmax[mask]
            out_rgb += (
                torch.exp(log_light_intensity).unsqueeze(-1)
                * 1
            )

        rgb_out_part = out_rgb
        all_rgb_out.append(rgb_out_part)
    all_rgb_out = torch.cat(all_rgb_out, dim=0)
    return all_rgb_out.view(img_params['height'], img_params['width'], -1)

if __name__ == "__main__":
    perfception_scannet_folder = "../PeRFception-ScanNet"
    original_scannet_folder = "../ScanNet/scans"
    scene_name = "plenoxel_scannet_scene0000_00"
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

    os.makedirs(render_dir, exist_ok=True)
    im = render_img(scene=scene, cam_params=cam_params, img_params=img_params)


    vis_vox(links_idx= scene.links_idx, density=scene.density)


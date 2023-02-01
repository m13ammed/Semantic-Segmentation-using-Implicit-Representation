import json
import os
from typing import *

import gin
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

import model.plenoxel_torch.dataclass as dataclass
import model.plenoxel_torch.sparse_grid as sparse_grid
import model.plenoxel_torch.utils as utils
import utils.ray as ray
import utils.store_util as store_util
from model.interface import LitModel
from model.plenoxel_torch.__global__ import BASIS_TYPE_SH




@gin.configurable()
class LitPlenoxel(LitModel): # a cleaned up version of the plenoxels model. keeping only parts ruiqurd for init teh model laoding and doing inference 

    # The external dataset will be called.
    def __init__(
        self,
        reso: List[List[int]] = [[256, 256, 256], [512, 512, 512]],
        upsample_step: List[int] = [38400, 76800],
        init_iters: int = 0,
        upsample_density_add: float = 0.0,
        basis_type: str = "sh",
        sh_dim: int = 9,
        mlp_posenc_size: int = 4,
        mlp_width: int = 32,
        background_nlayers: int = 0,
        background_reso: int = 512,
        # Sigma Optim
        sigma_optim: str = "rmsprop",
        lr_sigma: float = 3e1,
        lr_sigma_final: float = 5e-2,
        lr_sigma_decay_steps: int = 250000,
        lr_sigma_delay_steps: int = 15000,
        lr_sigma_delay_mult: float = 1e-2,
        # SH Optim
        sh_optim: str = "rmsprop",
        lr_sh: float = 1e-2,
        lr_sh_final: float = 5e-6,
        lr_sh_decay_steps: int = 250000,
        lr_sh_delay_steps: int = 0,
        lr_sh_delay_mult: float = 1e-2,
        lr_fg_begin_step: int = 0,
        # BG Simga Optim
        bg_optim: str = "rmsprop",
        lr_sigma_bg: float = 3e0,
        lr_sigma_bg_final: float = 3e-3,
        lr_sigma_bg_decay_steps: int = 250000,
        lr_sigma_bg_delay_steps: int = 0,
        lr_sigma_bg_delay_mult: float = 1e-2,
        # BG Colors Optim
        lr_color_bg: float = 1e-1,
        lr_color_bg_final: float = 5e-6,
        lr_color_bg_decay_steps: int = 250000,
        lr_color_bg_delay_steps: int = 0,
        lr_color_bg_delay_mult: float = 1e-2,
        # Basis Optim
        basis_optim: str = "rmsprop",
        lr_basis: float = 1e-6,
        lr_basis_final: float = 1e-6,
        lr_basis_decay_steps: int = 250000,
        lr_basis_delay_steps: int = 0,
        lr_basis_begin_step: int = 0,
        lr_basis_delay_mult: float = 1e-2,
        # RMSProp Option
        rms_beta: float = 0.95,
        # Init Option
        init_sigma: float = 0.1,
        init_sigma_bg: float = 0.1,
        thresh_type: str = "weight",
        weight_thresh: float = 0.0005 * 512,
        density_thresh: float = 5.0,
        background_density_thresh: float = 1.0 + 1e-9,
        max_grid_elements: int = 44_000_000,
        tune_mode: bool = False,
        tune_nosave: bool = False,
        # Losses
        lambda_tv: float = 1e-5,
        tv_sparsity: float = 0.01,
        tv_logalpha: bool = False,
        lambda_tv_sh: float = 1e-3,
        tv_sh_sparsity: float = 0.01,
        lambda_tv_lumisphere: float = 0.0,
        tv_lumisphere_sparsity: float = 0.01,
        tv_lumisphere_dir_factor: float = 0.0,
        tv_decay: float = 1.0,
        lambda_l2_sh: float = 0.0,
        tv_early_only: int = 1,
        tv_contiguous: int = 1,
        # Other Lambdas
        lambda_sparsity: float = 0.0,
        lambda_beta: float = 0.0,
        lambda_tv_background_sigma: float = 1e-2,
        lambda_tv_background_color: float = 1e-2,
        tv_background_sparsity: float = 0.01,
        # WD
        weight_decay_sigma: float = 1.0,
        weight_decay_sh: float = 1.0,
        lr_decay: bool = True,
        n_train: Optional[int] = None,
        nosphereinit: bool = False,
        # Render Options
        step_size: float = 0.5,
        sigma_thresh: float = 1e-8,
        stop_thresh: float = 1e-7,
        background_brightness: float = 1.0,
        renderer_backend: str = "cuvol",
        random_sigma_std: float = 0.0,
        random_sigma_std_background: float = 0.0,
        near_clip: float = 0.00,
        use_spheric_clip: bool = False,
        enable_random: bool = False,
        last_sample_opaque: bool = False,
        # Quantization
        filter_threshold: float = 0.0,
        quantize: bool = False,
        quantize_density: bool = False,
        store_efficient: bool = False,
        quant_bit: int = 8,
        logarithmic_quant: bool = False,
        clip_quant: bool = False,
        sh_clip_min: float = -0.1,
        sh_clip_max: float = 0.1,
        bkgd_clip_min: float = -4.0,
        bkgd_clip_max: float = 4.0,
        density_clip_min: float = 0,
        density_clip_max: float = 100,
        # Render Option
        bkgd_only: bool = False,
        # Scannet specific option
        init_grid_with_pcd: bool = True,
        upsample_stride: int = 1,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(LitPlenoxel, self).__init__()
        assert basis_type in ["sh", "3d_texture", "mlp"]
        assert sigma_optim in ["sgd", "rmsprop"]
        assert sh_optim in ["sgd", "rmsprop"]
        assert bg_optim in ["sgd", "rmsprop"]
        assert basis_optim in ["sgd", "rmsprop"]
        assert thresh_type in ["weight", "sigma"]
        assert renderer_backend in ["cuvol", "svox1", "nvol"]

        self.automatic_optimization = False
        self.reso_idx = 0
        self.reso_list = reso
        self.lr_sigma_func = self.get_expon_lr_func(
            lr_sigma,
            lr_sigma_final,
            lr_sigma_delay_steps,
            lr_sigma_delay_mult,
            lr_sigma_decay_steps,
        )
        self.lr_sh_func = self.get_expon_lr_func(
            lr_sh,
            lr_sh_final,
            lr_sh_delay_steps,
            lr_sh_delay_mult,
            lr_sh_decay_steps,
        )
        self.lr_sigma_bg_func = self.get_expon_lr_func(
            lr_sigma_bg,
            lr_sigma_bg_final,
            lr_sigma_bg_delay_steps,
            lr_sigma_bg_delay_mult,
            lr_sigma_bg_decay_steps,
        )
        self.lr_color_bg_func = self.get_expon_lr_func(
            lr_color_bg,
            lr_color_bg_final,
            lr_color_bg_delay_steps,
            lr_color_bg_delay_mult,
            lr_color_bg_decay_steps,
        )

    def setup(self, stage: Optional[str] = None) -> None:

        dmodule = self.trainer.datamodule

        self.model = sparse_grid.SparseGrid(
            reso=self.reso_list[self.reso_idx],
            center=dmodule.scene_center,
            radius=dmodule.scene_radius,
            use_sphere_bound=dmodule.use_sphere_bound and not self.nosphereinit,
            basis_dim=self.sh_dim,
            use_z_order=True,
            basis_type=eval("BASIS_TYPE_" + self.basis_type.upper()),
            mlp_posenc_size=self.mlp_posenc_size,
            mlp_width=self.mlp_width,
            background_nlayers=self.background_nlayers,
            background_reso=self.background_reso,
            device=self.device,
        )

        if stage is None or stage == "fit":
            self.model.sh_data.data[:] = 0.0
            self.model.density_data.data[:] = (
                0.0 if self.lr_fg_begin_step > 0 else self.init_sigma
            )
            if self.model.use_background:
                self.model.background_data.data[..., -1] = self.init_sigma_bg

        #if self.init_grid_with_pcd:
        #    self.initialize_with_pointcloud()

        self.ndc_coeffs = dmodule.ndc_coeffs

        with open(os.path.join(self.logdir, "config.gin"), "w") as f:
            f.write(gin.operative_config_str())

        return super().setup(stage)


    def generate_camera_list(
        self, intrinsics=None, extrinsics=None, ndc_coeffs=None, image_size=None
    ):
        dmodule = self.trainer.datamodule
        return [
            dataclass.Camera(
                torch.from_numpy(
                    self.extrinsics[i] if extrinsics is None else extrinsics[i]
                ).to(dtype=torch.float32, device=self.device),
                self.intrinsics[0, 0] if intrinsics is None else intrinsics[i, 0, 0],
                self.intrinsics[1, 1] if intrinsics is None else intrinsics[i, 1, 1],
                self.intrinsics[0, 2] if intrinsics is None else intrinsics[i, 0, 2],
                self.intrinsics[1, 2] if intrinsics is None else intrinsics[i, 1, 2],
                self.w if image_size is None else image_size[i, 1],
                self.h if image_size is None else image_size[i, 0],
                self.ndc_coeffs if ndc_coeffs is None else ndc_coeffs[i],
            )
            for i in dmodule.i_train
        ]

    def get_expon_lr_func(
        self, lr_init, lr_final, lr_delay_steps, lr_delay_mult, max_steps
    ):
        def helper(step):
            if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
                # Disable this parameter
                return 0.0
            if lr_delay_steps > 0:
                # A kind of reverse cosine decay.
                delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                    0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
                )
            else:
                delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        return helper

    def configure_optimizers(self):
        return None

    def quantize_data(self, data, eps=1e-12, clip_min=-0.1, clip_max=0.1):

        if self.quant_bit == 16:
            quant_data = data.type(torch.cuda.HalfTensor)
            return quant_data, 0., 1.

        if self.clip_quant:
            data = torch.clip(data, clip_min, clip_max)
            data_min, data_max = clip_min, clip_max
        else:
            data_min, data_max = (
                data.min(dim=0, keepdim=True)[0],
                data.max(dim=0, keepdim=True)[0],
            )

        data_scale = (data_max - data_min) / (2 ** self.quant_bit - 1)
        round_data = ((data - data_min) / (data_scale + eps)).round()

        if self.quant_bit == 8: 
            quant_data = round_data.type(torch.cuda.ByteTensor)

        elif self.quant_bit == 4:
            if len(round_data) % 2 == 1:
                round_data = torch.cat([round_data, torch.zeros(1, *round_data.shape[1:], device=round_data.device)], dim=0)
            round_data = round_data[0::2] * 16 + round_data[1::2] 
            quant_data = round_data.type(torch.cuda.ByteTensor)

        elif self.quant_bit == 2:
            dummy = 0 if len(round_data) % 4 == 0 else 4 - len(round_data) % 4
            if dummy != 0:
                round_data = torch.cat([round_data, torch.zeros(dummy, *round_data.shape[1:], device=round_data.device)], dim=0)
            round_data = round_data[0::4] * 64 + round_data[1::4] * 16 + round_data[2::4] * 4 + round_data[3::4]
            quant_data = round_data.type(torch.cuda.ByteTensor)

        return quant_data, data_min, data_scale

    def dequantize_data(self, data, data_min, data_scale):

        if self.quant_bit == 8 or self.quant_bit == 16: 
            data_tensor = data.type(torch.FloatTensor) * data_scale + data_min
        elif self.quant_bit == 4:
            data_blank = torch.zeros(len(data) * 2, *data.shape[1:], device=data.device)
            data_blank[0::2] = data // 16
            data_blank[1::2] = data % 16
            if torch.all(data_blank[-1] == 0): 
                data_blank = data_blank[:-1]
            data_tensor = data_blank.type(torch.FloatTensor) * data_scale + data_min
        elif self.quant_bit == 2:
            data_blank = torch.zeros(len(data) * 4, *data.shape[1:], device=data.device)
            data_blank[0::4] = data // 64
            data_blank[1::4] = data % 64 // 16
            data_blank[2::4] = data % 16 // 4
            data_blank[3::4] = data % 4
            for _ in range(4):
                if torch.all(data_blank[-1]) == 0:
                    data_blank = data_blank[:-1]
            data_tensor = data_blank.type(torch.FloatTensor) * data_scale + data_min

        if self.logarithmic_quant:
            data_tensor = torch.exp(-data_tensor)

        return data_tensor

    def on_predict_start(self) -> None:
        if self.bkgd_only:
            self.model.density_data.data[:] = -1e3
        return super().on_predict_start()

    def render_rays(
        self,
        batch,
        batch_idx,
        cpu=False,
        prefix="",
        randomize=False,
        render_bg=True,
        out_mask=False,
    ):
        ret = {}
        rays = batch["ray"].to(torch.float32)
        if "target" in batch.keys():
            target = batch["target"].to(torch.float32)
        else:
            target = (
                torch.zeros(
                    (len(batch["ray"]), 3), dtype=torch.float32, device=self.device
                )
                + 0.5
            )

        if self.ndc_coeffs[0] != -1 or self.ndc_coeffs[1] != -1:
            rays = torch.stack(
                ray.convert_to_ndc(rays[:, 0], rays[:, 1], self.ndc_coeffs), dim=1
            )

        rays = dataclass.Rays(rays[:, 0].contiguous(), rays[:, 1].contiguous())
        tup = self.model.volume_render_fused(
            rays,
            target,
            beta_loss=self.lambda_beta,
            sparsity_loss=self.lambda_sparsity,
            randomize=randomize,
            render_fg=True,
            render_bg=render_bg,
        )
        if len(tup) == 2:
           rgb, mask = tup
           sh = None
        else:
            rgb, mask, sh = tup
        # depth = self.model.volume_render_depth(
        #     rays,
        #     self.model.opt.sigma_thresh,
        # )
        if cpu:
            rgb = rgb.detach().cpu()
            if sh is not None:
                sh = sh.detach().cpu()
            # depth = depth.detach().cpu()
            target = target.detach().cpu()
            mask = mask.detach().cpu()

        rgb_key, target_key, mask_key, sh_key = "rgb", "target", "mask", "sh"
        if prefix != "":
            rgb_key = f"{prefix}/{rgb_key}"
            target_key = f"{prefix}/{target_key}"

        ret[rgb_key] = rgb
        # ret[depth_key] = depth[:, None]
        if out_mask and not self.bkgd_only:
            ret[mask_key] = mask
        if "target" in batch.keys():
            ret[target_key] = target
        if sh is not None:
            ret[sh_key] = sh
        return ret

    def predict_step(self, batch, batch_idx):
        ret = {}
        if self.bkgd_only:
            bg = self.render_rays(
                batch, batch_idx, cpu=True, prefix="bg", render_bg=True
            )
            ret.update(bg)
        elif self.trainer.datamodule.__class__.__name__=="LitDataPefceptionScannet": #if using out perfception data compute fg only
            fg = self.render_rays(
                batch,
                batch_idx,
                cpu=False,
                prefix="fg",
                render_bg=False,
            )
            ret.update(fg)
        else:
            fgbg = self.render_rays(
                batch,
                batch_idx,
                cpu=True,
                prefix="fgbg",
                render_bg=True,
                out_mask=True,
            )
            fg = self.render_rays(
                batch,
                batch_idx,
                cpu=True,
                prefix="fg",
                render_bg=False,
            )
            ret.update(fgbg)
            ret.update(fg)
        return ret

    @gin.configurable(module='LitPlenoxel')
    def on_predict_epoch_end(self, outputs, out_sh = False, output_dir = None):
        # In the prediction step, be sure to use outputs[0]
        # instead of outputs.
        render_poses = self.trainer.datamodule.render_poses
        N_img = len(render_poses)
        frame_ids = self.trainer.datamodule.trans_info['frame_ids']
        image_sizes = np.stack(
            [self.trainer.datamodule.image_sizes[0] for _ in range(N_img)]
        )
        if hasattr(self.trainer.datamodule, "render_scale"):
            image_sizes = (image_sizes * self.trainer.datamodule.render_scale).astype(
                image_sizes.dtype
            )

        keys = ["bg/rgb"] if self.bkgd_only else ["fgbg/rgb", "fg/rgb", "mask"]
        if self.trainer.datamodule.__class__.__name__=="LitDataPefceptionScannet": #if using our dataloadaer
            keys = ["fg/rgb"]
            if out_sh:
                keys.append("sh") #append the sh in addditon to the rgb_fg
        rets = {}
        for key in keys:
            ret = self.alter_gather_cat(outputs[0], key, image_sizes)
            rets[key] = ret

        if self.trainer.is_global_zero:
            os.makedirs("render", exist_ok=True)
            path_to_store = self.trainer.model.logdir
            scene_number = "_".join(path_to_store.split("_")[-3:])
            if self.trainer.datamodule.__class__.__name__ == "LitDataCo3D":
                with open("dataloader/co3d_lists/co3d_list.json") as fp:
                    co3d_list = json.load(fp)
                class_name = co3d_list[scene_number]
                class_path = f"render/{class_name}"
                scene_path = f"render/{class_name}/{scene_number}"
            else:
                scene_name = self.trainer.datamodule.scene_name
                scene_path = f"render/{scene_name}"
            opt_list = ["bg"] if self.bkgd_only else ["fg", "fgbg"]

            os.makedirs(scene_path, exist_ok=True)
            if self.trainer.datamodule.__class__.__name__=="LitDataPefceptionScannet":
                #opt_path = os.path.join(scene_path, 'fg')
                #os.makedirs(opt_path, exist_ok=True)
                if output_dir is not None:
                    scene_path = f"{scene_name}"
                    opt_path = os.path.join(output_dir,scene_path)
                    os.makedirs(opt_path, exist_ok=True)
                else:
                    opt_path = scene_path
                store_util.store_image_pose_num(opt_path, rets[f"fg/rgb"], frame_ids) #save rgb images
                if out_sh:
                    store_util.store_sh_pose_num(opt_path, rets[f"sh"], frame_ids) #save sh harmonics as npz files
            else:
                for opt in opt_list:
                    opt_path = os.path.join(scene_path, opt)
                    os.makedirs(opt_path, exist_ok=True)
                    store_util.store_image(opt_path, rets[f"{opt}/rgb"])
                    

            if "mask" in rets.keys():
                mask_path = os.path.join(scene_path, "mask")
                os.makedirs(mask_path, exist_ok=True)
                store_util.store_mask(mask_path, rets["mask"])

        #if self.bkgd_only:
        np.save(f"{scene_path}/perf_poses.npy", self.trainer.datamodule.render_poses) #save the poses in perf format (transformed to be aligned with the sparse grid)
        np.save(f"{scene_path}/intrinsics.npy", self.trainer.datamodule.intrinsics) #save intriscis of camera
        np.save(f"{scene_path}/poses.npy", self.trainer.datamodule.extrinsics) #save the poses in original fomrat

    def on_save_checkpoint(self, checkpoint) -> None:

        checkpoint["reso_idx"] = self.reso_idx
        density_data = checkpoint["state_dict"]["model.density_data"].cpu()
        if self.quantize_density: 
            density_data, density_min, density_scale = self.quantize_data(
                density_data, clip_min=self.density_clip_min, clip_max=self.density_clip_max
            )

        sh = checkpoint["state_dict"]["model.sh_data"]
        if self.quantize:
            sh, sh_min, sh_scale = self.quantize_data(sh, clip_min=self.sh_clip_min, clip_max=self.sh_clip_max)

        model_links = checkpoint["state_dict"]["model.links"].cpu()
        reso_list = self.reso_list[self.reso_idx]

        argsort_val = model_links[torch.where(model_links >= 0)].argsort()
        links_compressed = torch.stack(torch.where(model_links >= 0))[:, argsort_val]
        links_idx = (
            reso_list[1] * reso_list[2] * links_compressed[0]
            + reso_list[2] * links_compressed[1]
            + links_compressed[2]
        )

        links = -torch.ones_like(model_links, device="cpu")
        links[
            links_compressed[0], links_compressed[1], links_compressed[2]
        ] = torch.arange(len(links_compressed[0]), dtype=torch.int32, device="cpu")

        background_data = checkpoint["state_dict"]["model.background_data"].cpu()

        if self.model.use_background:
            if self.quantize:
                background_data, bkgd_min, bkgd_scale = self.quantize_data(
                    background_data, clip_min=self.bkgd_clip_min, clip_max=self.bkgd_clip_max
                )
                checkpoint["model.background_data_min"] = bkgd_min
                checkpoint["model.background_data_scale"] = bkgd_scale
            checkpoint["state_dict"].pop("model.background_data")

        checkpoint["state_dict"]["model.background_data"] = background_data

        checkpoint["state_dict"].pop("model.density_data")
        checkpoint["state_dict"].pop("model.sh_data")
        checkpoint["state_dict"].pop("model.links")

        checkpoint["state_dict"]["model.density_data"] = density_data   

        if self.quantize_density: 
            checkpoint["model.density_data_min"] = density_min
            checkpoint["model.density_data_scale"] = density_scale

        checkpoint["state_dict"]["model.links_idx"] = links_idx
        checkpoint["state_dict"]["model.sh_data"] = sh
        if self.quantize:
            checkpoint["model.sh_data_min"] = sh_min
            checkpoint["model.sh_data_scale"] = sh_scale
            if self.model.use_background:
                checkpoint["model.background_data_min"] = bkgd_min
                checkpoint["model.background_data_scale"] = bkgd_scale

        return super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint) -> None:

        state_dict = checkpoint["state_dict"]

        self.reso_idx = checkpoint["reso_idx"]

        del self.model.basis_data
        del self.model.density_data
        del self.model.sh_data
        del self.model.links

        self.model.register_parameter(
            "basis_data", nn.Parameter(state_dict["model.basis_data"])
        )

        if "model.background_data_min" in checkpoint.keys():
            del self.model.background_data
            bgd_data = state_dict["model.background_data"]
            if self.quantize:
                bgd_min = checkpoint["model.background_data_min"]
                bgd_scale = checkpoint["model.background_data_scale"]
                bgd_data = self.dequantize_data(bgd_data, bgd_min, bgd_scale)

            self.model.register_parameter("background_data", nn.Parameter(bgd_data))
            checkpoint["state_dict"]["model.background_data"] = bgd_data

        density_data = state_dict["model.density_data"]
        if self.quantize_density:
            density_min = checkpoint["model.density_data_min"]
            density_scale = checkpoint["model.density_data_scale"]
            density_data = self.dequantize_data(density_data, density_min, density_scale)

        self.model.register_parameter("density_data", nn.Parameter(density_data))
        checkpoint["state_dict"]["model.density_data"] = density_data

        sh_data = state_dict["model.sh_data"]
        if self.quantize:
            sh_data_min = checkpoint["model.sh_data_min"]
            sh_data_scale = checkpoint["model.sh_data_scale"]
            sh_data = self.dequantize_data(sh_data, sh_data_min, sh_data_scale)

        self.model.register_parameter("sh_data", nn.Parameter(sh_data))
        checkpoint["state_dict"]["model.sh_data"] = sh_data

        reso = self.reso_list[checkpoint["reso_idx"]]

        links = torch.zeros(reso, dtype=torch.int32) - 1
        links_sparse = state_dict["model.links_idx"]
        links_idx = torch.stack(
            [
                links_sparse // (reso[1] * reso[2]),
                links_sparse % (reso[1] * reso[2]) // reso[2],
                links_sparse % reso[2],
            ]
        ).long()
        links[links_idx[0], links_idx[1], links_idx[2]] = torch.arange(
            len(links_idx[0]), dtype=torch.int32
        )
        checkpoint["state_dict"].pop("model.links_idx")
        checkpoint["state_dict"]["model.links"] = links
        self.model.register_buffer("links", links)

        return super().on_load_checkpoint(checkpoint)

import argparse
import logging
import os
import shutil
from typing import *

import gin
import pytorch_lightning.loggers as pl_loggers
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.plugins import DDPPlugin
from dataloader.litdata import LitDataPefceptionScannet
#from utils.logger import RetryingWandbLogger
from model.plenoxel_torch.model import LitPlenoxel


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise Exception("Boolean value expected.")


@gin.configurable()
def run(
    datadir: Optional[str] = None,
    logbase: Optional[str] = None,
    scene_name: Optional[str] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = "perfception_scannet",
    postfix: Optional[str] = None,
    # Optimization
    max_steps: int = 200000,
    precision: int = 32,
    # Logging
    log_every_n_steps: int = 1000,
    progressbar_refresh_rate: int = 5,
    # Run Mode
    run_render: bool = True,
    accelerator: str = "gpu",
    num_gpus: Optional[int] = 1,
    num_tpus: Optional[int] = None,
    num_sanity_val_steps: int = 0,
    seed: int = 777,
    debug: bool = False,
    save_last_only: bool = False,
    check_val_every_n_epoch: int = 1,
):

    if scene_name is None and dataset_name == "perfception_scannet":
            scene_name = "plenoxel_scannet_scene0000_00"
    print(scene_name, dataset_name, "\n\n\n")
    exp_name =  dataset_name + "_" + scene_name


    seed_everything(seed, workers=True)

    callbacks = [TQDMProgressBar(refresh_rate=progressbar_refresh_rate)]

    trainer = Trainer(
        log_every_n_steps=log_every_n_steps,
        devices=num_gpus,
        max_steps=max_steps,
        replace_sampler_ddp=False,
        check_val_every_n_epoch=check_val_every_n_epoch,
        precision=precision,
        accelerator="gpu",
        num_sanity_val_steps=num_sanity_val_steps,
        callbacks = callbacks
    )


    data_module = LitDataPefceptionScannet(
        datadir=datadir, 
        scene_name=scene_name, 
        accelerator=accelerator,
        num_gpus=num_gpus,
        num_tpus=num_tpus,
    )
    model = LitPlenoxel()
    os.makedirs("/log/",exist_ok=True)
    model.logdir = "/log/"


    if run_render:
        checkpoont_path = os.path.join(datadir,scene_name,"last.ckpt")
        trainer.predict(model, data_module, ckpt_path= checkpoont_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="scene name",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )

    args = parser.parse_args()

    ginbs = []
    logging.info(f"Gin configuration files: {args.ginc}")
    gin.parse_config_files_and_bindings(args.ginc,args.ginb)
    run(
        scene_name=args.scene_name,
    )


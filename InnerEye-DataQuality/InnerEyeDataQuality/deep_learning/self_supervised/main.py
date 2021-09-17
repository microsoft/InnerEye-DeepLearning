#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Union

import pytorch_lightning as pl
import torch

from default_paths import EXPERIMENT_DIR
from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.self_supervised.byol.byol_module import BYOLInnerEye
from InnerEyeDataQuality.deep_learning.self_supervised.datamodules.utils import create_ssl_data_modules
from InnerEyeDataQuality.deep_learning.self_supervised.simclr_module import SimCLRInnerEye
from InnerEyeDataQuality.deep_learning.self_supervised.ssl_classifier_module import (SSLOnlineEvaluatorInnerEye,
                                                                                     get_encoder_output_dim)
from InnerEyeDataQuality.deep_learning.utils import load_ssl_model_config
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

num_gpus = torch.cuda.device_count()
num_devices = num_gpus if num_gpus > 0 else 1
trained_kwargs: Dict[str, Union[str, int, float]] = {"precision": 16} if num_gpus > 0 else {}
if num_gpus > 1:  # If multi-gpu training update the parameters for DDP
    trained_kwargs.update({"distributed_backend": "ddp", "sync_batchnorm": True})

def get_last_checkpoint_path(default_root_dir: str, model_version: str) -> str:
    return str(Path(default_root_dir) / model_version / "checkpoints" / "last.ckpt")


def cli_main(config: ConfigNode, debug: bool = False) -> None:
    """
    Runs self-supervised training on imaging data using contrastive loss.
    Currently it supports only ``BYOL`` and ``SimCLR``.

    :param config: A ssl_model config specifying training configurations (and augmentations).
                   Beware the augmentations parameters are ignored at the moment when using CIFAR10 as we
                   always use the default augmentations from PyTorch Lightning.
    :param debug: If set to True, only runs training and validation on 1% of the data.
    """

    # set seed
    seed_everything(config.train.seed)

    # self-supervision type
    ssl_type = config.train.self_supervision.type
    default_root_dir = EXPERIMENT_DIR / config.train.output_dir
    model_version = f'{ssl_type}_seed_{config.train.seed}'
    checkpoint_dir = str(default_root_dir / model_version / "checkpoints")

    # Model checkpointing callback
    checkpoint_callback = ModelCheckpoint(period=config.train.checkpoint_period, save_top_k=-1, dirpath=checkpoint_dir)
    checkpoint_callback_last = ModelCheckpoint(save_last=True, dirpath=checkpoint_dir)

    lr_logger = LearningRateMonitor()
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=str(default_root_dir),
                                             version='logs',
                                             name=model_version)
    # Create SimCLR data modules and model
    dm = create_ssl_data_modules(config)
    if ssl_type == "simclr":
        model = SimCLRInnerEye(num_samples=dm.num_samples,  # type: ignore
                               batch_size=dm.batch_size,  # type: ignore
                               lr=config.train.base_lr,
                               dataset_name=config.dataset.name,
                               encoder_name=config.train.self_supervision.encoder_name)
    # Create BYOL model
    else:
        model = BYOLInnerEye(num_samples=dm.num_samples,  # type: ignore
                             learning_rate=config.train.base_lr,
                             dataset_name=config.dataset.name,
                             encoder_name=config.train.self_supervision.encoder_name,
                             batch_size=dm.batch_size,  # type: ignore
                             warmup_epochs=10)
    model.hparams.update({'ssl_type': ssl_type})

    # Online fine-tunning using an MLP
    online_eval = SSLOnlineEvaluatorInnerEye(class_weights=dm.class_weights,  # type: ignore
                                             z_dim=get_encoder_output_dim(model, dm),
                                             num_classes=dm.num_classes,  # type: ignore
                                             dataset=config.dataset.name,
                                             drop_p=0.2)  # type: ignore

    # Load latest checkpoint
    resume_from_last_checkpoint = get_last_checkpoint_path(default_root_dir, model_version) if \
        config.train.resume_from_last_checkpoint else None
    if debug:
        overfit_batches = num_devices / (min(len(dm.val_dataloader()), len(dm.train_dataloader())) * 2.0)
        trained_kwargs.update({"overfit_batches": overfit_batches})

    # Create trainer and run training
    trainer = pl.Trainer(gpus=num_gpus,
                         logger=tb_logger,
                         default_root_dir=str(default_root_dir),
                         benchmark=True,
                         max_epochs=config.scheduler.epochs,
                         callbacks=[lr_logger, online_eval, checkpoint_callback, checkpoint_callback_last],
                         resume_from_checkpoint=resume_from_last_checkpoint,
                         **trained_kwargs)
    trainer.fit(model, dm)  # type: ignore


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    parser = argparse.ArgumentParser(description='Train a self-supervised model')
    parser.add_argument('--config', dest='config', type=str, required=True,
                        help='Path to config file characterising trained CNN model/s')
    args, unknown_args = parser.parse_known_args()
    config_path = args.config
    config = load_ssl_model_config(config_path)
    # Launch the script
    cli_main(config)


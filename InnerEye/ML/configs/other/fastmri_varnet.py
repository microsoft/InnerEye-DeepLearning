#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa
from typing import Optional

import param
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.tensorboard import SummaryWriter

from InnerEye.Common.common_util import add_folder_to_sys_path_if_needed
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference

add_folder_to_sys_path_if_needed("fastMRI")

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule


class VarNetWithImageLogging(VarNetModule):
    """
    A clone of the VarNet model that logs images to only the Tensorboard loggers. The original VarNet hardcodes
    a single logger that must be Tensorboard.
    """

    def log_image(self, name: str, image: torch.Tensor) -> None:
        experiments = self.logger.experiment if isinstance(self.logger.experiment, list) \
            else [self.logger.experiment]
        for experiment in experiments:
            if isinstance(experiment, SummaryWriter):
                experiment.add_image(name, image, global_step=self.global_step)


class FastMri(LightningContainer):
    # All fields that are declared here will be automatically available as commandline arguments.
    challenge: str = param.String(default="multicoil", doc="Chooses between the singlecoil or multicoil"
                                                           "acquisition setup.")
    sample_rate: Optional[float] = param.Number(default=None, doc="Fraction of slices of the training data split to "
                                                                  "use. Default: 1.0")

    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id = "fastmrimini_brain"

    def create_model(self) -> LightningWithInference:
        return VarNetWithImageLogging()

    def get_data_module(self) -> LightningDataModule:
        mask = create_mask_for_mask_type(mask_type_str="equispaced",
                                         center_fractions=[0.08],
                                         accelerations=[4])
        # use random masks for train transform, fixed masks for val transform
        train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
        val_transform = VarNetDataTransform(mask_func=mask)
        test_transform = VarNetDataTransform()

        return FastMriDataModule(data_path=self.local_dataset,
                                 challenge=self.challenge,
                                 sample_rate=self.sample_rate,
                                 train_transform=train_transform,
                                 val_transform=val_transform,
                                 test_transform=test_transform)

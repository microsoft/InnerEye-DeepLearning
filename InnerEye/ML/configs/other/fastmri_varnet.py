#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa
import logging
import shutil
from pathlib import Path
from typing import Optional

import param
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.tensorboard import SummaryWriter

from InnerEye.Common.common_util import add_folder_to_sys_path_if_needed
from InnerEye.ML.lightning_container import LightningContainer

add_folder_to_sys_path_if_needed("fastMRI")

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule, VarNetModule

# The name of the dataset cache file that the fastMRI codebase generates
DATASET_CACHE = "dataset_cache.pkl"


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


def get_data_module(azure_dataset_id: str,
                    local_dataset: Optional[Path],
                    sample_rate: Optional[float],
                    test_path: str) -> LightningDataModule:
    """
    Creates a LightningDataModule that consumes data from the FastMRI challenge. The type of challenge
    (single/multicoil) is determined from the name of the dataset in Azure blob storage. The mask type is set to
    equispaced, with 4x acceleration.
    :param azure_dataset_id: The name of the dataset (folder name in blob storage).
    :param local_dataset: The local folder at which the dataset has been mounted or downloaded.
    :param sample_rate: Fraction of slices of the training data split to use. Set to a value <1.0 for rapid prototyping.
    :param test_path: The name of the folder inside the dataset that contains the test data.
    :return: A LightningDataModule object.
    """
    if not azure_dataset_id:
        raise ValueError("The azure_dataset_id argument must be provided.")
    if not local_dataset:
        raise ValueError("The local_dataset argument must be provided.")
    for challenge in ["multicoil", "singlecoil"]:
        if challenge in azure_dataset_id:
            break
    else:
        raise ValueError(f"Unable to determine the value for the challenge field for this "
                         f"dataset: {azure_dataset_id}")

    mask = create_mask_for_mask_type(mask_type_str="equispaced",
                                     center_fractions=[0.08],
                                     accelerations=[4])
    # use random masks for train transform, fixed masks for val transform
    train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
    val_transform = VarNetDataTransform(mask_func=mask)
    test_transform = VarNetDataTransform()

    return FastMriDataModule(data_path=local_dataset,
                             test_path=local_dataset / test_path,
                             challenge=challenge,
                             sample_rate=sample_rate,
                             train_transform=train_transform,
                             val_transform=val_transform,
                             test_transform=test_transform)


class FastMri(LightningContainer):
    """
    A base class for all models for the FastMRI challenge. It implements the `create_model` overload, but not
    the `get_data_module` method.
    """
    # All fields that are declared here will be automatically available as commandline arguments.
    sample_rate: Optional[float] = param.Number(default=None, doc="Fraction of slices of the training data split to "
                                                                  "use. Default: 1.0")

    def __init__(self) -> None:
        super().__init__()
        self.num_epochs = 50
        self.pl_progress_bar_refresh_rate = 100

    def create_model(self) -> LightningModule:
        return VarNetWithImageLogging()

    def before_training_on_local_rank_zero(self) -> None:
        assert self.local_dataset, "No dataset is available yet."
        dataset_cache = self.local_dataset / DATASET_CACHE
        if dataset_cache.is_file():
            # There is no easy way of overriding the location of the dataset cache file in the
            # constructor of FastMriDataModule. Hence, copy from dataset folder to current working directory.
            logging.info("Copying the dataset cache file to the current working directory.")
            shutil.copy(dataset_cache, Path.cwd() / DATASET_CACHE)
        else:
            logging.info("No dataset cache file found in the dataset folder.")


class KneeMulticoil(FastMri):
    """
    A model configuration to train a VarNet model on the knee_multicoil dataset, with 4x acceleration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id = "knee_multicoil"

    def get_data_module(self) -> LightningDataModule:
        return get_data_module(azure_dataset_id=self.azure_dataset_id,
                               local_dataset=self.local_dataset,
                               sample_rate=self.sample_rate,
                               test_path="multicoil_test_v2")


class KneeSinglecoil(FastMri):
    """
    A model configuration to train a VarNet model on the knee_singlecoil dataset, with 4x acceleration.
    """

    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id = "knee_singlecoil"

    def get_data_module(self) -> LightningDataModule:
        return get_data_module(azure_dataset_id=self.azure_dataset_id,
                               local_dataset=self.local_dataset,
                               sample_rate=self.sample_rate,
                               test_path="singlecoil_test_v2")


class BrainMulticoil(FastMri):
    """
    A model configuration to train a VarNet model on the brain_multicoil dataset, with 4x acceleration.
    When training this model, it is possible that the cloud nodes run out of disk space when downloading the dataset.
    If this happens, supply the additional "--use_dataset_mount=True" on the commandline when submitting the job.
    """

    def __init__(self) -> None:
        super().__init__()
        self.azure_dataset_id = "brain_multicoil"
        # If the Azure nodes run out of disk space when downloading the dataset, re-submit with the
        # --use_dataset_mount=True flag. The dataset will be mounted to the fixed path given here.
        self.dataset_mountpoint = "/tmp/fastmri"

    def get_data_module(self) -> LightningDataModule:
        return get_data_module(azure_dataset_id=self.azure_dataset_id,
                               local_dataset=self.local_dataset,
                               sample_rate=self.sample_rate,
                               test_path="multicoil_test")

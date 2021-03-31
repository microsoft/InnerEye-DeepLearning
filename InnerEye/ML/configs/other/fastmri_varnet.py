#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
from _pytest.monkeypatch import MonkeyPatch
from pytorch_lightning import LightningDataModule

from InnerEye.Common.common_util import add_folder_to_sys_path_if_needed
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference

add_folder_to_sys_path_if_needed("fastMRI")

from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import VarNetModule
from fastmri.pl_modules import FastMriDataModule
from tests.create_temp_data import create_temp_data


class VarNetWithInference(LightningWithInference,
                          VarNetModule):

    def inference_step(self, batch: Any, batch_idx: int, model_output: torch.Tensor):
        pass


class FastMriRandomData(FastMriDataModule):
    def __init__(self, data_path: Path):
        if data_path.is_dir():
            shutil.rmtree(str(data_path))
        data_path.mkdir(exist_ok=False, parents=True)
        _, _, metadata = create_temp_data(data_path)

        def retrieve_metadata_mock(a, fname):
            return metadata[str(fname)]

        # That's a bit flaky, we should be un-doing that after, but there's no obvious place of doing so.
        MonkeyPatch().setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

        mask = create_mask_for_mask_type(mask_type_str="equispaced",
                                         center_fractions=[0.08],
                                         accelerations=[4])
        # use random masks for train transform, fixed masks for val transform
        train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
        val_transform = VarNetDataTransform(mask_func=mask)
        test_transform = VarNetDataTransform()

        FastMriDataModule.__init__(self,
                                   data_path=data_path / "knee_data",
                                   challenge="multicoil",
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform)

    def prepare_data(self, *args, **kwargs):
        print("FastMriRandomData.prepare_data")

    def setup(self, stage: Optional[str] = None):
        print("FastMriRandomData.setup")


class FastMriDemo(LightningContainer):
    def __init__(self):
        super().__init__()
        self.num_epochs = 1
        # Restrict to a single GPU, because we tthat could cause race conditions
        self.max_num_gpus = 1

    def create_model(self) -> LightningWithInference:
        return VarNetModule()

    def get_data_module(self) -> LightningDataModule:
        # Local_dataset is set via the commandline to a random folder for unit testss
        return FastMriRandomData(data_path=self.local_dataset)

# Invoke via: runner.py --model FastMriDemo

# Things to add: type checks in loader. Is the model derived from LightningWithInference? Derived from LightningModule?
# Get the code that uses .fit back in.

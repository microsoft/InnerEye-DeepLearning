#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import torch
from pytorch_lightning import LightningDataModule

from InnerEye.Common.common_util import add_folder_to_sys_path_if_needed
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference

add_folder_to_sys_path_if_needed("fastMRI")

from fastmri.pl_modules import VarNetModule


class VarNetWithInference(LightningWithInference,
                          VarNetModule):
    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        pass

    def inference_step(self, batch: Any, batch_idx: int, model_output: torch.Tensor):
        pass

    def on_inference_epoch_end(self) -> None:
        pass


class FastMriDemoContainer(LightningContainer):

    def create_model(self) -> LightningWithInference:
        return VarNetWithInference()

    def get_data_module(self) -> LightningDataModule:
        return 1

# Invoke via: runner.py --model FastMriDemoContainer

# Things to add: type checks in loader. Is the model derived from LightningWithInference? Derived from LightningModule?
# Get the code that uses .fit back in.

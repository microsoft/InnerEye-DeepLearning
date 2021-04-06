#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import torch

from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.SSL.datamodules.cxr_datasets import RSNAKaggleCXR
from InnerEye.SSL.datamodules.datamodules import InnerEyeVisionDataModule

from InnerEye.SSL.utils import load_ssl_model_config


def test_weights_innereye_module() -> None:
    """
    Tests if weights in data module are correctly initialized
    """
    dataset_dir = str(Path(__file__).parent / "test_dataset")
    data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                  return_index=False,
                                  train_transforms=None,
                                  val_transforms=None,
                                  data_dir=str(dataset_dir),
                                  batch_size=25,
                                  seed=1)
    data_module.setup()
    assert torch.isclose(data_module.compute_class_weights(), torch.tensor([0.20, 0.80], dtype=torch.float32),
                         atol=1e-3).all()

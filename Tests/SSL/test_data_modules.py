#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import torch

from InnerEye.ML.SSL.datamodules.cxr_datasets import RSNAKaggleCXR
from InnerEye.ML.SSL.datamodules.datamodules import InnerEyeVisionDataModule


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
    class_weights = data_module.compute_class_weights()
    assert class_weights is not None
    assert torch.isclose(class_weights, torch.tensor([0.20, 0.80], dtype=torch.float32), atol=1e-3).all()

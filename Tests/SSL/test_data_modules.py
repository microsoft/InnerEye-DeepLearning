#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch

from InnerEye.ML.SSL.datamodules_and_datasets.cxr_datasets import RSNAKaggleCXR
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import InnerEyeVisionDataModule
from Tests.SSL.test_ssl_containers import _create_test_cxr_data
from Tests.utils_for_tests import full_ml_test_data_path


def test_weights_innereye_module() -> None:
    """
    Tests if weights in data module are correctly initialized
    """
    path_to_test_dataset = full_ml_test_data_path("cxr_test_dataset")
    _create_test_cxr_data(path_to_test_dataset)
    data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                           return_index=False,
                                           train_transforms=None,
                                           val_transforms=None,
                                           data_dir=str(path_to_test_dataset),
                                           batch_size=25,
                                           seed=1)
    data_module.setup()
    class_weights = data_module.compute_class_weights()
    assert class_weights is not None
    assert torch.isclose(class_weights, torch.tensor([0.21, 0.79], dtype=torch.float32), atol=1e-3).all()

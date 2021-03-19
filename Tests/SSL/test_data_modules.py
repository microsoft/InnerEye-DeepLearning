#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import torch

from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.SSL.datamodules.chestxray_datamodule import RSNAKaggleDataModule
from InnerEye.SSL.utils import load_ssl_model_config


def test_weights_rsna_module() -> None:
    """
    Tests if weights in data module are correctly initialized
    """

    config = load_ssl_model_config(repository_root_directory() / "InnerEye" / "SSL" / "configs" / "rsna_byol.yaml")
    config.defrost()
    config.train.self_supervision.use_balanced_binary_loss_for_linear_head = False
    config.dataset.dataset_dir = str(Path(__file__).parent / "test_dataset")
    data_module = RSNAKaggleDataModule(config, num_devices=1, num_workers=1)  # type: ignore
    assert data_module.class_weights is None

    config.train.self_supervision.use_balanced_binary_loss_for_linear_head = True
    data_module = RSNAKaggleDataModule(config, num_devices=1, num_workers=1)  # type: ignore
    assert data_module.class_weights is not None
    assert torch.isclose(data_module.class_weights, torch.tensor([0.2208, 0.7792], dtype=torch.float32),
                         atol=1e-3).all()

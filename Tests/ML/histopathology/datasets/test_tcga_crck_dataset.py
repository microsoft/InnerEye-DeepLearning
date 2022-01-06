#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os

import pytest
import torch
from monai.data.dataset import Dataset
from torch.utils.data import DataLoader

from InnerEye.ML.Histopathology.datasets.default_paths import TCGA_CRCK_DATASET_DIR
from InnerEye.ML.Histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from InnerEye.ML.Histopathology.models.transforms import LoadTiled


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk dataset is unavailable")
@pytest.mark.parametrize('train', [True, False])
def test_dataset(train: bool) -> None:
    base_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR, train=train)
    dataset = Dataset(base_dataset, transform=LoadTiled('image'))  # type: ignore

    expected_length = 93408 if train else 98904
    assert len(dataset) == expected_length

    sample = dataset[0]
    expected_keys = ['slide_id', 'tile_id', 'image', 'split', 'label']
    assert all(key in sample for key in expected_keys)
    assert isinstance(sample['image'], torch.Tensor)
    assert sample['image'].shape == (3, 224, 224)

    batch_size = 16
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # type: ignore
    batch = next(iter(loader))
    assert all(key in batch for key in expected_keys)
    assert isinstance(batch['image'], torch.Tensor)
    assert batch['image'].shape == (batch_size, 3, 224, 224)
    assert batch['image'].dtype == torch.float32
    assert batch['label'].shape == (batch_size,)
    assert batch['label'].dtype == torch.int64

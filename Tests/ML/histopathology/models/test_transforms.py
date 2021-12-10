#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path
from typing import Callable, Sequence, Union

import pytest
import torch
from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from monai.transforms import Compose
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import Subset
from torchvision.models import resnet18

from health_ml.utils.bag_utils import BagDataset
from InnerEye.ML.Histopathology.datasets.default_paths import TCGA_CRCK_DATASET_DIR
from InnerEye.ML.Histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from InnerEye.ML.Histopathology.models.encoders import ImageNetEncoder
from InnerEye.ML.Histopathology.models.transforms import EncodeTilesBatchd, LoadTiled, LoadTilesBatchd
from Tests.ML.util import assert_dicts_equal


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_load_tile() -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    load_transform = LoadTiled(image_key)
    index = 0

    # Test that the transform affects only the image entry in the sample
    input_sample = tiles_dataset[index]
    loaded_sample = load_transform(input_sample)
    assert_dicts_equal(loaded_sample, input_sample, exclude_keys=[image_key])

    # Test that the MONAI Dataset applies the same transform
    loaded_dataset = Dataset(tiles_dataset, transform=load_transform)  # type:ignore
    same_dataset_sample = loaded_dataset[index]
    assert_dicts_equal(same_dataset_sample, loaded_sample)

    # Test that loading another sample gives different results
    different_sample = loaded_dataset[index + 1]
    assert not torch.allclose(different_sample[image_key], loaded_sample[image_key])


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_load_tiles_batch() -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    max_bag_size = 5
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)
    load_batch_transform = LoadTilesBatchd(image_key)
    loaded_dataset = Dataset(tiles_dataset, transform=LoadTiled(image_key))  # type:ignore
    image_shape = loaded_dataset[0][image_key].shape
    index = 0

    # Test that the transform affects only the image entry in the batch,
    # and that the loaded images have the expected shape
    bagged_batch = bagged_dataset[index]
    manually_loaded_batch = load_batch_transform(bagged_batch)
    assert_dicts_equal(manually_loaded_batch, bagged_batch, exclude_keys=[image_key])
    assert manually_loaded_batch[image_key].shape == (max_bag_size, *image_shape)

    # Test that the MONAI Dataset applies the same transform
    loaded_bagged_dataset = Dataset(bagged_dataset, transform=load_batch_transform)  # type:ignore
    loaded_bagged_batch = loaded_bagged_dataset[index]
    assert_dicts_equal(loaded_bagged_batch, manually_loaded_batch)

    # Test that loading another batch gives different results
    different_batch = loaded_bagged_dataset[index + 1]
    assert not torch.allclose(different_batch[image_key], manually_loaded_batch[image_key])

    # Test that loading and bagging commute
    bagged_loaded_dataset = BagDataset(loaded_dataset,  # type: ignore
                                       bag_ids=tiles_dataset.slide_ids,
                                       max_bag_size=max_bag_size)
    bagged_loaded_batch = bagged_loaded_dataset[index]
    assert_dicts_equal(bagged_loaded_batch, loaded_bagged_batch)


def _test_cache_and_persistent_datasets(tmp_path: Path,
                                        base_dataset: TorchDataset,
                                        transform: Union[Sequence[Callable], Callable],
                                        cache_subdir: str) -> None:
    default_dataset = Dataset(base_dataset, transform=transform)  # type: ignore
    cached_dataset = CacheDataset(base_dataset, transform=transform)  # type: ignore
    cache_dir = tmp_path / cache_subdir
    cache_dir.mkdir(exist_ok=True)
    persistent_dataset = PersistentDataset(base_dataset, transform=transform,  # type: ignore
                                           cache_dir=cache_dir)

    for default_sample, cached_sample, persistent_sample \
            in zip(default_dataset, cached_dataset, persistent_dataset):  # type: ignore
        assert_dicts_equal(cached_sample, default_sample)
        assert_dicts_equal(persistent_sample, default_sample)


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
def test_cached_loading() -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN

    max_num_tiles = 100
    tiles_subset = Subset(tiles_dataset, range(max_num_tiles))
    _test_cache_and_persistent_datasets(tiles_subset, transform=LoadTiled(image_key),
                                        cache_subdir="TCGA-CRCk_tiles_cache")

    max_bag_size = 5
    max_num_bags = max_num_tiles // max_bag_size
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)
    bagged_subset = Subset(bagged_dataset, range(max_num_bags))
    _test_cache_and_persistent_datasets(bagged_subset, transform=LoadTilesBatchd(image_key),
                                        cache_subdir="TCGA-CRCk_load_cache")


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk tiles dataset is unavailable")
@pytest.mark.parametrize('use_gpu', [False, True])
def test_encode_tiles(use_gpu: bool) -> None:
    tiles_dataset = TcgaCrck_TilesDataset(TCGA_CRCK_DATASET_DIR)
    image_key = tiles_dataset.IMAGE_COLUMN
    max_bag_size = 5
    bagged_dataset = BagDataset(tiles_dataset, bag_ids=tiles_dataset.slide_ids,  # type: ignore
                                max_bag_size=max_bag_size)

    encoder = ImageNetEncoder(resnet18, tile_size=224, n_channels=3)
    if use_gpu:
        encoder.cuda()

    encode_transform = EncodeTilesBatchd(image_key, encoder)
    transform = Compose([LoadTilesBatchd(image_key), encode_transform])
    dataset = Dataset(bagged_dataset, transform=transform)  # type: ignore
    sample = dataset[0]
    assert sample[image_key].shape == (max_bag_size, encoder.num_encoding)
    # TODO: Ensure it works in DDP

    max_num_bags = 20
    bagged_subset = Subset(bagged_dataset, range(max_num_bags))
    _test_cache_and_persistent_datasets(bagged_subset, transform=transform,
                                        cache_subdir="TCGA-CRCk_embed_cache")

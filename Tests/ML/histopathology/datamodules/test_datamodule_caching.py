import shutil
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from InnerEye.ML.Histopathology.datamodules.base_module import CacheMode, TilesDataModule
from InnerEye.ML.Histopathology.datasets.base_dataset import TilesDataset


def noop_transform(x: Any) -> Any:
    return x


def _check_generator_consistency(dl: DataLoader) -> None:
    dataloader_generator = dl.generator
    bag_sampler_generator = dl.dataset.data.bag_sampler.generator  # type: ignore
    assert torch.equal(dataloader_generator.get_state(),
                       bag_sampler_generator.get_state())


def compare_dataloaders(dl1: DataLoader, dl2: DataLoader) -> None:
    for batch1, batch2 in zip(dl1, dl2):
        _check_generator_consistency(dl1)
        _check_generator_consistency(dl2)
        assert batch1.keys() == batch2.keys()
        for key in batch1:
            assert len(batch1[key]) == len(batch2[key])
            for item1, item2 in zip(batch1[key], batch2[key]):
                if isinstance(item1, torch.Tensor):
                    assert torch.allclose(item1, item2, equal_nan=True)
                else:
                    assert item1 == item2


class MockTilesDataset(TilesDataset):
    TILE_X_COLUMN = TILE_Y_COLUMN = None
    TRAIN_SPLIT_LABEL = 'train'
    VAL_SPLIT_LABEL = 'val'
    TEST_SPLIT_LABEL = 'test'


def generate_mock_dataset_df(n_slides: int, n_tiles: int, n_classes: int) -> pd.DataFrame:
    slide_ids = np.random.randint(n_slides, size=n_tiles)
    slide_labels = np.random.randint(n_classes, size=n_slides)
    tile_labels = slide_labels[slide_ids]
    split_labels = [MockTilesDataset.TRAIN_SPLIT_LABEL,
                    MockTilesDataset.VAL_SPLIT_LABEL,
                    MockTilesDataset.TEST_SPLIT_LABEL]
    slide_splits = np.random.choice(split_labels, size=n_slides)
    tile_splits = slide_splits[slide_ids]

    df = pd.DataFrame()
    df[MockTilesDataset.TILE_ID_COLUMN] = np.arange(n_tiles)
    df[MockTilesDataset.SLIDE_ID_COLUMN] = slide_ids
    df[MockTilesDataset.LABEL_COLUMN] = tile_labels
    df[MockTilesDataset.SPLIT_COLUMN] = tile_splits
    df[MockTilesDataset.IMAGE_COLUMN] = [f"{tile_splits[i]}/{i:06d}.png" for i in range(n_tiles)]

    return df


class MockTilesDataModule(TilesDataModule):
    def get_splits(self) -> Tuple[MockTilesDataset, MockTilesDataset, MockTilesDataset]:
        df = MockTilesDataset(self.root_path).dataset_df
        df = df.reset_index()
        split_dfs = (df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TRAIN_SPLIT_LABEL],
                     df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.VAL_SPLIT_LABEL],
                     df[df[MockTilesDataset.SPLIT_COLUMN] == MockTilesDataset.TEST_SPLIT_LABEL])
        return tuple(MockTilesDataset(self.root_path, dataset_df=split_df)  # type: ignore
                     for split_df in split_dfs)


@pytest.fixture
def mock_data_dir(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "mock_tiles_dataset"
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / MockTilesDataset.DEFAULT_CSV_FILENAME
    if not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = generate_mock_dataset_df(n_slides=8, n_tiles=100, n_classes=2)
        df.to_csv(csv_path, index=False)
    return csv_dir

def _get_datamodule(cache_mode: CacheMode, save_precache: bool,
                    cache_dir_provided: bool, data_dir: Path) -> TilesDataModule:
    if (cache_mode is CacheMode.NONE and save_precache) \
            or (cache_mode is CacheMode.DISK and not cache_dir_provided) \
            or (save_precache and not cache_dir_provided):
        pytest.skip("Unsupported combination of caching arguments")

    cache_dir = data_dir / f"datamodule_cache_{cache_mode.value}" if cache_dir_provided else None

    if cache_dir is not None and cache_dir.exists():
        shutil.rmtree(cache_dir)

    return MockTilesDataModule(root_path=data_dir,
                               transform=noop_transform,
                               seed=0,
                               batch_size=2,
                               cache_mode=cache_mode,
                               save_precache=save_precache,
                               cache_dir=cache_dir)


@pytest.mark.parametrize('cache_mode', [CacheMode.MEMORY, CacheMode.DISK, CacheMode.NONE])
@pytest.mark.parametrize('save_precache', [True, False])
@pytest.mark.parametrize('cache_dir_provided', [True, False])
def test_caching_consistency(mock_data_dir: Path, cache_mode: CacheMode, save_precache: bool,
                             cache_dir_provided: bool) -> None:
    # Compare two dataloaders from the same datamodule
    datamodule = _get_datamodule(cache_mode=cache_mode,
                                 save_precache=save_precache,
                                 cache_dir_provided=cache_dir_provided,
                                 data_dir=mock_data_dir)
    datamodule.prepare_data()
    train_dataloader = datamodule.train_dataloader()
    train_dataloader2 = datamodule.train_dataloader()

    compare_dataloaders(train_dataloader, train_dataloader2)

    # Compare datamodules reusing the same cache
    datamodule = _get_datamodule(cache_mode=cache_mode,
                                 save_precache=save_precache,
                                 cache_dir_provided=cache_dir_provided,
                                 data_dir=mock_data_dir)
    datamodule.prepare_data()
    train_dataloader = datamodule.train_dataloader()

    reloaded_datamodule = _get_datamodule(cache_mode=cache_mode,
                                          save_precache=save_precache,
                                          cache_dir_provided=cache_dir_provided,
                                          data_dir=mock_data_dir)
    reloaded_datamodule.prepare_data()
    reloaded_train_dataloader = reloaded_datamodule.train_dataloader()

    compare_dataloaders(train_dataloader, reloaded_train_dataloader)


@pytest.mark.parametrize('cache_mode', [CacheMode.MEMORY, CacheMode.DISK, CacheMode.NONE])
@pytest.mark.parametrize('save_precache', [True, False])
@pytest.mark.parametrize('cache_dir_provided', [True, False])
def test_tile_id_coverage(mock_data_dir: Path, cache_mode: CacheMode, save_precache: bool,
                          cache_dir_provided: bool) -> None:
    datamodule = _get_datamodule(cache_mode=cache_mode,
                                 save_precache=save_precache,
                                 cache_dir_provided=cache_dir_provided,
                                 data_dir=mock_data_dir)
    datamodule.prepare_data()
    train_dataset = datamodule.train_dataset
    train_dataloader = datamodule.train_dataloader()
    expected_tile_ids = set(train_dataset.dataset_df.index)
    loaded_tile_ids = set()  # type: ignore
    for batch in train_dataloader:
        for stacked_bag_tile_ids in batch[train_dataset.TILE_ID_COLUMN]:
            if isinstance(stacked_bag_tile_ids, torch.Tensor):
                stacked_bag_tile_ids = stacked_bag_tile_ids.tolist()
            bag_tile_ids = set(stacked_bag_tile_ids)
            assert bag_tile_ids.isdisjoint(loaded_tile_ids), \
                f"Tile IDs already seen: {bag_tile_ids}"
            loaded_tile_ids.update(bag_tile_ids)
    assert loaded_tile_ids == expected_tile_ids

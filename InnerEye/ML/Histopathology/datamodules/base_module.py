#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from monai.data.dataset import CacheDataset, Dataset, PersistentDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from health_ml.utils.bag_utils import BagDataset, multibag_collate
from health_ml.utils.common_utils import _create_generator
from InnerEye.ML.Histopathology.datasets.base_dataset import TilesDataset
from InnerEye.ML.Histopathology.models.transforms import LoadTilesBatchd


class CacheMode(Enum):
    NONE = 'none'
    MEMORY = 'memory'
    DISK = 'disk'

class CacheLocation(Enum):
    NONE = 'none'
    CPU = 'cpu'
    GPU = 'cuda'
class TilesDataModule(LightningDataModule):
    """Base class to load the tiles of a dataset as train, val, test sets"""

    def __init__(self, root_path: Path, max_bag_size: int = 0, batch_size: int = 1,
                 seed: Optional[int] = None, transform: Optional[Callable] = None,
                 cache_mode: CacheMode = CacheMode.NONE,
                 precache_location: CacheLocation = CacheLocation.NONE,
                 cache_dir: Optional[Path] = None,
                 number_of_cross_validation_splits: int = 0,
                 cross_validation_split_index: int = 0) -> None:
        """
        :param root_path: Root directory of the source dataset.
        :param max_bag_size: Upper bound on number of tiles in each loaded bag. If 0 (default),
        will return all samples in each bag. If > 0 , bags larger than `max_bag_size` will yield
        random subsets of instances.
        :param batch_size: Number of slides to load per batch.
        :param seed: pseudorandom number generator seed to use for shuffling instances and bags. Note that randomness in
        train/val/test splits is handled independently in `get_splits()`. (default: `None`)
        :param transform: A transform to apply to the source tiles dataset, or a composition of
        transforms using `monai.transforms.Compose`. By default (`None`), applies `LoadTilesBatchd`.
        :param cache_mode: The type of caching to perform, i.e. whether the results of all
        transforms up to the first randomised one should be computed only once and reused in
        subsequent iterations:
          - `MEMORY`: MONAI CacheDataset is used, the entire transformed dataset is kept in memory for fastest access;
          - `DISK`: MONAI PersistentDataset is used, each transformed sample is saved to disk and loaded on-demand;
          - `NONE` (default): standard MONAI dataset is used, no caching is performed.
        :param precache_location: Whether to pre-cache the entire transformed dataset upfront and save
        it to disk. This is done once in `prepare_data()` only on the local rank-0 process, so
        multiple processes can afterwards access the same cache without contention in DDP settings. This parameter also allow to
        choose if the cache will be re-loaded into CPU or GPU memory:
          - `NONE (default)`: no pre-cache is performed;
          - `CPU`: each transformed sample is saved to disk and, if cache_mode is `MEMORY`, reloaded into CPU;
          - `GPU`: each transformed sample is saved to disk and, if cache_mode is `MEMORY`, reloaded into GPU memory;
        If cache_mode is `DISK` precache_location `CPU` and `GPU` are equivalent.
        :param cache_dir: The directory onto which to cache data if caching is enabled.
        :param number_of_cross_validation_splits: Number of folds to perform.
        :param cross_validation_split_index: Index of the cross validation split to be performed.
        """
        if precache_location is not CacheLocation.NONE and cache_mode is CacheMode.NONE:
            raise ValueError("Can only pre-cache if caching is enabled")
        if precache_location is not CacheLocation.NONE and cache_dir is None:
            raise ValueError("A cache directory is required for pre-caching")
        if cache_mode is CacheMode.DISK and cache_dir is None:
            raise ValueError("A cache directory is required for on-disk caching")
        super().__init__()

        self.root_path = root_path
        self.max_bag_size = max_bag_size
        self.transform = transform
        self.cache_mode = cache_mode
        self.precache_location = precache_location
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.number_of_cross_validation_splits = number_of_cross_validation_splits
        self.cross_validation_split_index = cross_validation_split_index
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_splits()
        self.class_weights = self.train_dataset.get_class_weights()
        self.seed = seed

    def get_splits(self) -> Tuple[TilesDataset, TilesDataset, TilesDataset]:
        """Create the training, validation, and test datasets"""
        raise NotImplementedError

    def prepare_data(self) -> None:
        if self.precache_location != CacheLocation.NONE:
            self._load_dataset(self.train_dataset, stage='train', shuffle=True)
            self._load_dataset(self.val_dataset, stage='val', shuffle=True)
            self._load_dataset(self.test_dataset, stage='test', shuffle=True)

    def _dataset_pickle_path(self, stage: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{stage}_dataset.pt"

    def _load_dataset(self, tiles_dataset: TilesDataset, stage: str, shuffle: bool) -> Dataset:
        dataset_pickle_path = self._dataset_pickle_path(stage)

        if dataset_pickle_path and dataset_pickle_path.is_file():
            # torch.load will reload on GPU by default, same device it was saved from
            memory_location = torch.device('cpu') if self.precache_location == CacheLocation.CPU else None
            with dataset_pickle_path.open('rb') as f:
                print(f"Loading dataset from {dataset_pickle_path} into {memory_location}")
                return torch.load(f, map_location=memory_location)

        generator = _create_generator(self.seed)
        bag_dataset = BagDataset(tiles_dataset,  # type: ignore
                                 bag_ids=tiles_dataset.slide_ids,
                                 max_bag_size=self.max_bag_size,
                                 shuffle_samples=shuffle,
                                 generator=generator)
        transform = self.transform or LoadTilesBatchd(tiles_dataset.IMAGE_COLUMN)

        # Save and restore PRNG state for consistency across (pre-)caching options
        generator_state = generator.get_state()
        transformed_bag_dataset = self._get_transformed_dataset(bag_dataset, transform)  # type: ignore
        generator.set_state(generator_state)

        # Dataset is saved if cache_dir is True, regardless of CacheMode
        if dataset_pickle_path:
            dataset_pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with dataset_pickle_path.open('wb') as f:
                torch.save(transformed_bag_dataset, f)

        return transformed_bag_dataset

    def _get_transformed_dataset(self, base_dataset: BagDataset,
                                 transform: Union[Sequence[Callable], Callable]) -> Dataset:
        if self.cache_mode is CacheMode.MEMORY:
            dataset = CacheDataset(base_dataset, transform, num_workers=1)  # type: ignore
        elif self.cache_mode is CacheMode.DISK:
            dataset = PersistentDataset(base_dataset, transform, cache_dir=self.cache_dir)  # type: ignore
            if self.precache_location != CacheLocation.NONE:
                import tqdm  # TODO: Make optional

                for i in tqdm.trange(len(dataset), desc="Loading dataset"):
                    dataset[i]  # empty loop to pre-compute all transformed samples
        else:
            dataset = Dataset(base_dataset, transform)  # type: ignore
        return dataset

    def _get_dataloader(self, tiles_dataset: TilesDataset, stage: str, shuffle: bool,
                        **dataloader_kwargs: Any) -> DataLoader:
        transformed_bag_dataset = self._load_dataset(tiles_dataset, stage=stage, shuffle=shuffle)
        bag_dataset: BagDataset = transformed_bag_dataset.data  # type: ignore
        generator = bag_dataset.bag_sampler.generator
        return DataLoader(transformed_bag_dataset, batch_size=self.batch_size,
                          collate_fn=multibag_collate, shuffle=shuffle, generator=generator,
                          pin_memory=False,  # disable pinning as loaded data may already be on GPU
                          **dataloader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, 'train', shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, 'val', shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, 'test', shuffle=True)

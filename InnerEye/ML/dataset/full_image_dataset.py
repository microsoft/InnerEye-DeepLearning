#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from abc import ABC
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

import pandas as pd
import torch.utils.data
from torch._six import container_abcs
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import default_collate  # type: ignore

from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.sample import GeneralSampleMetadata, PatientDatasetSource, \
    PatientMetadata, Sample
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils import io_util, ml_util
from InnerEye.ML.utils.csv_util import CSV_CHANNEL_HEADER, CSV_PATH_HEADER, \
    CSV_SUBJECT_HEADER
from InnerEye.ML.utils.transforms import Compose3D

COMPRESSION_EXTENSIONS = ['sz', 'gz']


def collate_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    The collate function that the dataloader workers should use. It does the same thing for all "normal" fields
    (all fields are put into tensors with outer dimension batch_size), except for the special "metadata" field.
    Those metadata objects are collated into a simple list.
    :param batch: A list of samples that should be collated.
    :return: collated result
    """
    elem = batch[0]
    if isinstance(elem, container_abcs.Mapping):
        result = dict()
        for key in elem:
            # Special handling for all fields that store metadata, and for fields that are list.
            # Lists are used in SequenceDataset.
            # All these are collated by turning them into lists or lists of lists.
            if isinstance(elem[key], (list, PatientMetadata, GeneralSampleMetadata)):
                result[key] = [d[key] for d in batch]
            else:
                result[key] = default_collate([d[key] for d in batch])
        return result
    raise TypeError(f"Unexpected batch data: Expected a dictionary, but got: {type(elem)}")


def set_random_seed_for_dataloader_worker(worker_id: int) -> None:
    """
    Set the seed for the random number generators of python, numpy.
    """
    # Set the seeds for numpy and python random based on the offset of the worker_id and initial seed,
    # converting the initial_seed which is a long to modulo int32 which is what numpy expects.
    random_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    ml_util.set_random_seed(random_seed, f"Data loader worker ({worker_id})")


class _RepeatSampler(BatchSampler):
    """
    A batch sampler that wraps another batch sampler. It repeats the contents of that other sampler forever.
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False, max_repeats: int = 0) -> None:
        super().__init__(sampler, batch_size, drop_last)
        self.sampler = sampler
        self.max_repeats = max_repeats

    def __iter__(self) -> Any:
        repeats = 0
        while self.max_repeats == 0 or repeats < self.max_repeats:
            yield from iter(self.sampler)
            repeats += 1


class ImbalancedSampler(Sampler):
    """
    Sampler that performs naive over-sampling by drawing samples with
    replacements. The probability of being drawn depends on the label of
    each data point, rare labels have a higher probability to be drawn.
    Assumes the dataset implements the "get_all_labels" functions in order
    to compute the weights associated with each data point.

    Side note: the sampler choice is independent from the data augmentation
    pipeline. Data augmentation is performed on the images while loading them
    at a later stage. This sampler merely affects which item is selected.
    """

    # noinspection PyMissingConstructor
    def __init__(self, dataset: Any, num_samples: int = None) -> None:
        """

        :param dataset: a dataset
        :num_samples: number of samples to draw. If None the number of samples
        corresponds to the length of the dataset.
        """
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        self.weights = self.get_weights()
        self.num_samples = len(dataset) if num_samples is None else num_samples

    def get_weights(self) -> torch.Tensor:
        labels = self.dataset.get_labels_for_imbalanced_sampler()
        counts_per_label: Dict = Counter(labels)
        return torch.tensor([1.0 / counts_per_label[labels[i]] for i in self.indices])

    def __iter__(self) -> Any:
        # noinspection PyTypeChecker
        return iter([self.indices[i] for i in torch.multinomial(self.weights, self.num_samples,  # type: ignore
                                                                replacement=True)])

    def __len__(self) -> int:
        return self.num_samples


class RepeatDataLoader(DataLoader):
    """
    This class implements a data loader that avoids spawning a new process after each epoch.
    It uses an infinite sampler.
    This is adapted from https://github.com/pytorch/pytorch/issues/15849
    """

    def __init__(self,
                 dataset: Any,
                 max_repeats: int,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 use_imbalanced_sampler: bool = False,
                 drop_last: bool = False,
                 **kwargs: Any):
        """
        Creates a new data loader.
        :param dataset: The dataset that should be loaded.
        :param batch_size: The number of samples per minibatch.
        :param shuffle: If true, the dataset will be shuffled randomly.
        :param drop_last: If true, drop incomplete minibatches at the end.
        :param kwargs: Additional arguments that will be passed through to the Dataloader constructor.
        """
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        if use_imbalanced_sampler:
            sampler = ImbalancedSampler(dataset)
        self._actual_batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        repeat_sampler = _RepeatSampler(self._actual_batch_sampler, batch_size=batch_size, max_repeats=max_repeats)
        super().__init__(dataset=dataset, batch_sampler=repeat_sampler, **kwargs)
        self.iterator = None

    def __len__(self) -> int:
        return len(self._actual_batch_sampler)

    def __iter__(self) -> Any:
        if self.iterator is None:
            self.iterator = super().__iter__()  # type: ignore
        assert self.iterator is not None  # for mypy
        for i in range(len(self)):
            yield next(self.iterator)


D = TypeVar('D', bound=ModelConfigBase)


class GeneralDataset(Dataset, ABC, Generic[D]):
    def __init__(self, args: D, data_frame: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None):
        self.name = name or "None"
        self.args = args
        self.data_frame = args.dataset_data_frame if data_frame is None else data_frame
        logging.info(f"Processing dataset (name={self.name})")

    def as_data_loader(self,
                       shuffle: bool,
                       batch_size: Optional[int] = None,
                       num_dataload_workers: Optional[int] = None,
                       use_imbalanced_sampler: bool = False,
                       drop_last_batch: bool = False,
                       max_repeats: Optional[int] = None) -> DataLoader:
        num_dataload_workers = num_dataload_workers or self.args.num_dataload_workers
        batch_size = batch_size or self.args.train_batch_size
        if self.args.avoid_process_spawn_in_data_loaders:
            if max_repeats is None:
                max_repeats = self.args.get_total_number_of_training_epochs()
            return RepeatDataLoader(
                self,
                max_repeats=max_repeats,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_dataload_workers,
                pin_memory=self.args.pin_memory,
                worker_init_fn=set_random_seed_for_dataloader_worker,
                collate_fn=collate_with_metadata,
                use_imbalanced_sampler=use_imbalanced_sampler,
                drop_last=drop_last_batch
            )
        else:
            if use_imbalanced_sampler:
                sampler: Optional[Sampler] = ImbalancedSampler(self)
                shuffle = False
            else:
                sampler = None
            return DataLoader(
                self,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_dataload_workers,
                pin_memory=self.args.pin_memory,
                worker_init_fn=set_random_seed_for_dataloader_worker,
                collate_fn=collate_with_metadata,
                sampler=sampler,  # type: ignore
                drop_last=drop_last_batch
            )


class FullImageDataset(GeneralDataset):
    """
    Dataset class that loads and creates samples with full 3D images from a given pd.Dataframe. The following
    are the operations performed to generate a sample from this dataset:
    -------------------------------------------------------------------------------------------------
    1) On initialization parses the provided pd.Dataframe with dataset information, to cache the set of file paths
       and patient mappings to load as PatientDatasetSource. The sources are then saved in a list: dataset_sources.
    2) dataset_sources is iterated in a batched fashion, where for each batch it loads the full 3D images, and applies
       pre-processing functions (e.g. normalization), returning a sample that can be used for full image operations.
    """

    def __init__(self, args: SegmentationModelBase, data_frame: pd.DataFrame,
                 full_image_sample_transforms: Optional[Compose3D[Sample]] = None):
        super().__init__(args, data_frame)
        self.full_image_sample_transforms = full_image_sample_transforms

        # Check base_path
        assert self.args.local_dataset is not None
        if not self.args.local_dataset.is_dir():
            raise ValueError("local_dataset should be the path to the base directory of the data: {}".
                             format(self.args.local_dataset))

        # cache all of the available dataset sources
        dataloader: Callable[[], Any] = self._load_dataset_sources

        self.dataset_sources: Dict[str, PatientDatasetSource] = dataloader()
        self.dataset_indices: List[str] = sorted(self.dataset_sources.keys())

    def __len__(self) -> int:
        return len(self.dataset_indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return self.get_samples_at_index(index=i)[0].get_dict()

    @staticmethod
    def _extension_from_df_file_paths(file_paths: List[str]) -> str:
        file_extensions = [f.split('.')[-2] if f.endswith(tuple(COMPRESSION_EXTENSIONS))
                           else f.split('.')[-1] for f in file_paths]
        if len(file_extensions) == 0:
            raise Exception("No files of expected format (Nifti) were found")
        # files must all be of same type
        unique_file_extensions = list(set(file_extensions))
        if len(unique_file_extensions) > 1:
            raise Exception("More than one file type was found. This is not supported.")
        return "." + unique_file_extensions[0]

    def get_samples_at_index(self, index: int) -> List[Sample]:
        # load the channels into memory
        ds = self.dataset_sources[self.dataset_indices[index]]
        samples = [io_util.load_images_from_dataset_source(dataset_source=ds)]  # type: ignore
        return [Compose3D.apply(self.full_image_sample_transforms, x) for x in samples]

    def _load_dataset_sources(self) -> Dict[str, PatientDatasetSource]:
        assert self.args.local_dataset is not None
        return load_dataset_sources(dataframe=self.data_frame,
                                    local_dataset_root_folder=self.args.local_dataset,
                                    image_channels=self.args.image_channels,
                                    ground_truth_channels=self.args.ground_truth_ids,
                                    mask_channel=self.args.mask_id
                                    )


def load_dataset_sources(dataframe: pd.DataFrame,
                         local_dataset_root_folder: Path,
                         image_channels: List[str],
                         ground_truth_channels: List[str],
                         mask_channel: Optional[str]) -> Dict[str, PatientDatasetSource]:
    """
    Prepares a patient-to-images mapping from a dataframe read directly from a dataset CSV file.
    The dataframe contains per-patient per-channel image information, relative to a root directory.
    This method converts that into a per-patient dictionary, that contains absolute file paths
    separated for for image channels, ground truth channels, and mask channels.
    :param dataframe: A dataframe read directly from a dataset CSV file.
    :param local_dataset_root_folder: The root folder that contains all images.
    :param image_channels: The names of the image channels that should be used in the result.
    :param ground_truth_channels: The names of the ground truth channels that should be used in the result.
    :param mask_channel: The name of the mask channel that should be used in the result. This can be None.
    :return: A dictionary mapping from an integer subject ID to a PatientDatasetSource.
    """
    expected_headers = {CSV_SUBJECT_HEADER, CSV_PATH_HEADER, CSV_CHANNEL_HEADER}
    # validate the csv file
    actual_headers = list(dataframe)
    if not expected_headers.issubset(actual_headers):
        raise ValueError("The dataset CSV file should contain at least these columns: {}, but got: {}"
                         .format(expected_headers, actual_headers))

    # Calculate unique data points, first, and last data point
    unique_ids: List[str] = sorted(pd.unique(dataframe[CSV_SUBJECT_HEADER]))
    if not local_dataset_root_folder.is_dir():
        raise ValueError("The dataset root folder does not exist: {}".format(local_dataset_root_folder))

    def get_mask_channel_or_default() -> Optional[Path]:
        if mask_channel is None:
            return None
        else:
            return get_paths_for_channel_ids(channels=[mask_channel])[0]

    def get_paths_for_channel_ids(channels: List[str]) -> List[Path]:
        if len(set(channels)) < len(channels):
            raise ValueError(f"ids have duplicated entries: {channels}")

        paths: List[Path] = []
        rows = dataframe.loc[dataframe[CSV_SUBJECT_HEADER] == patient_id]
        for channel_id in channels:
            row = rows.loc[rows[CSV_CHANNEL_HEADER] == channel_id]
            if len(row) == 0:
                raise ValueError(f"Patient {patient_id} does not have channel '{channel_id}'")
            elif len(row) > 1:
                raise ValueError(f"Patient {patient_id} has more than one entry for channel '{channel_id}'")
            image_path = local_dataset_root_folder / row[CSV_PATH_HEADER].values[0]
            paths.append(image_path)
        return paths

    dataset_sources = {}
    for patient_id in unique_ids:
        metadata = PatientMetadata.from_dataframe(dataframe, patient_id)
        dataset_sources[patient_id] = PatientDatasetSource(
            metadata=metadata,
            image_channels=get_paths_for_channel_ids(channels=image_channels),  # type: ignore
            mask_channel=get_mask_channel_or_default(),
            ground_truth_channels=get_paths_for_channel_ids(channels=ground_truth_channels)  # type: ignore
        )

    return dataset_sources

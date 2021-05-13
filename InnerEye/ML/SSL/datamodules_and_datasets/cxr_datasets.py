#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Subset
from torchvision.datasets import VisionDataset

from InnerEye.Common.type_annotations import PathOrString
from InnerEye.ML.SSL.datamodules_and_datasets.dataset_cls_utils import InnerEyeDataClassBaseWithReturnIndex, \
    OptionalIndexInputAndLabel
from InnerEye.ML.utils.io_util import is_dicom_file_path, load_dicom_image


class InnerEyeCXRDatasetBase(VisionDataset):
    """
    Base class for a dataset with X-ray images and image-level target labels.
    Implements reading of dicom files as well as png.
    """

    def __init__(self,
                 root: str,
                 train: bool,
                 transform: Optional[Callable] = None,
                 **kwargs: Any) -> None:
        """

        :param root: path to the data directory
        :param train: if True returns the train + val dataset, if False returns the test set. See VisionDataset API.
        :param transform: callable to be applied on the loaded image, has to take PIL Image as input
        """
        super().__init__(root=root, transforms=transform)
        self.root = Path(self.root)  # type: ignore
        if not self.root.exists():
            logging.error(
                f"The data directory {self.root} does not exist. Make sure to download the data first.")
        self.train = train
        self.targets: Optional[List[int]] = None
        self._prepare_dataset()

    def _prepare_dataset(self) -> None:
        self.indices: List[int] = []
        self.filenames: List[PathOrString] = []
        raise NotImplementedError("_prepare_dataset needs to be implemented by the child classes.")

    def __getitem__(self, index: int) -> OptionalIndexInputAndLabel:
        """
        :param index: The index of the sample to be fetched
        :return: The image and (fake) label tensors
        """
        filename = self.filenames[index]
        target = self.targets[index] if self.targets is not None else 0
        if is_dicom_file_path(str(filename)):
            scan_image = load_dicom_image(filename)
            # Dicom files have arbitrary pixel intensities, convert to [0,255] range so that PIL can
            # read the array into an Image object.
            scan_image = (scan_image - scan_image.min()) * 255. / (scan_image.max() - scan_image.min())
            # Load as PIL Image in grey-scale (convert("L") step), yields a 1-channel image.
            scan_image = Image.fromarray(scan_image).convert("L")
        else:
            # Load as PIL Image in grey-scale (convert("L") step), yields a 1-channel image with pixel values in range
            # [0,1] (float).
            scan_image = Image.open(filename).convert("L")
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        return scan_image, target

    def __len__(self) -> int:
        """
        :return: The size of the dataset
        """
        return len(self.indices)


class InnerEyeCXRDatasetWithReturnIndex(InnerEyeDataClassBaseWithReturnIndex, InnerEyeCXRDatasetBase):
    """
    Any dataset used in SSL needs to inherit from InnerEyeDataClassBaseWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance.
    """
    pass


class RSNAKaggleCXR(InnerEyeCXRDatasetWithReturnIndex):
    """
    Dataset class to load the RSNA Chest-Xray training dataset.
    For more details instructions about how to download the dataset and the expected data folder structure, please
    refer to the docs/self_supervised_models.md documentation.

    Note we use all the data for train and val. No test data implemented.
    """

    def _prepare_dataset(self) -> None:
        if self.train:
            self.dataset_dataframe = pd.read_csv(self.root / "dataset.csv")
            self.targets = self.dataset_dataframe.label.values.astype(np.int64)
            self.subject_ids = self.dataset_dataframe.subject.values
            self.indices = np.arange(len(self.dataset_dataframe))
            self.filenames = [self.root / f"{subject_id}.dcm" for subject_id in self.subject_ids]
        else:
            # No test set implemented for this data class.
            self.indices = []
            self.filenames = []

    @property
    def num_classes(self) -> int:
        return 2


class NIHCXR(InnerEyeCXRDatasetWithReturnIndex):
    """
    Dataset class to load the NIHCXR Chest-Xray dataset.
    For more details instructions about how to download the dataset and the expected data folder structure, please
    refer to the docs/self_supervised_models.md documentation.

    Note: by default, the code uses the full data for training and validation (including
    the official test set). You can change this behavior by setting `use_full_dataset_for_train_and_val` to
    False when you initialize the class.
    """

    def __init__(self,
                 root: str,
                 use_full_dataset_for_train_and_val: bool = True,
                 **kwargs: Any) -> None:
        self.use_full_dataset_for_train_and_val = use_full_dataset_for_train_and_val
        super().__init__(root=root, **kwargs)

    def _prepare_dataset(self) -> None:
        self.dataset_dataframe = pd.read_csv(self.root / "Data_Entry_2017.csv")
        # To use full dataset (incl. official test set for train & val of SSL models, no test set)
        if self.use_full_dataset_for_train_and_val:
            self.subject_ids = self.dataset_dataframe["Image Index"].values if self.train else []
        # To exclude official test set from train & val
        else:
            train_ids = pd.read_csv(self.root / "train_val_list.txt", header=None).values.reshape(-1)
            is_train_val_ids = self.dataset_dataframe["Image Index"].isin(train_ids).values
            self.subject_ids = np.where(is_train_val_ids)[0] if self.train else np.where(~is_train_val_ids)[0]
        self.indices = np.arange(len(self.subject_ids))
        self.filenames = [self.root / f"{subject_id}" for subject_id in self.subject_ids]


class CheXpert(InnerEyeCXRDatasetWithReturnIndex):
    """
    Dataset class to load the CheXpert dataset.

    For more details instructions about how to download the dataset and the expected data folder structure, please
    refer to the docs/self_supervised_models.md documentation.

    Note: By default, we exclude lateral scans from the dataset, you can change this behavior by
    setting remove_lateral_scans_from_dataset to False when you initialize your dataset class.
    """

    def __init__(self,
                 root: str,
                 remove_lateral_scans_from_dataset: bool = True,
                 **kwargs: Any) -> None:
        self.remove_lateral_scans_from_dataset = remove_lateral_scans_from_dataset
        super().__init__(root=root, **kwargs)

    def _prepare_dataset(self) -> None:
        if self.train:  # for train AND val
            self.dataset_dataframe = pd.read_csv(self.root / "train.csv")
        else:  # test set (unused in SSL training)
            self.dataset_dataframe = pd.read_csv(self.root / "valid.csv")

        if self.remove_lateral_scans_from_dataset:  # Remove lateral shots
            self.dataset_dataframe = self.dataset_dataframe.loc[self.dataset_dataframe["Frontal/Lateral"] == "Frontal"]

        # Strip away the name of the folder that is included in the path column of the dataset
        strip_n = len("CheXpert-v1.0-small/")
        self.dataset_dataframe.Path = self.dataset_dataframe.Path.apply(lambda x: x[strip_n:])
        self.indices = np.arange(len(self.dataset_dataframe))
        self.filenames = [self.root / p for p in self.dataset_dataframe.Path.values]


class CovidDataset(InnerEyeCXRDatasetWithReturnIndex):
    """
    Dataset class to load CovidDataset dataset as datamodule for monitoring SSL training quality directly on
    CovidDataset data.
    We use CVX03 against CVX12 as proxy task.
    """

    def _prepare_dataset(self) -> None:
        self.dataset_dataframe = pd.read_csv(self.root / "dataset.csv")
        mapping = {"0": 0, "3": 0, "1": 1, "2": 1}
        self.dataset_dataframe["final_label"] = self.dataset_dataframe.final_label.astype(int).astype(str).apply(
            lambda x: mapping[x])
        self.indices = np.arange(len(self.dataset_dataframe))
        self.subject_ids = self.dataset_dataframe.subject.values
        self.filenames = [self.root / file for file in self.dataset_dataframe.filepath.values]
        self.targets = self.dataset_dataframe.final_label.values.astype(np.int64).reshape(-1)

    @property
    def num_classes(self) -> int:
        return 2

    def _split_dataset(self, val_split: float, seed: int) -> Tuple[Subset, Subset]:
        """
        Implements val - train split.
        :param val_split: proportion to use for validation
        :param seed: random seed for splitting
        :return: dataset_train, dataset_val
        """
        shuffled_subject_ids = np.random.RandomState(seed).permutation(np.unique(self.subject_ids))
        n_val = int(len(shuffled_subject_ids) * val_split)
        val_subjects, train_subjects = shuffled_subject_ids[:n_val], shuffled_subject_ids[n_val:]
        train_ids, val_ids = np.where(np.isin(self.subject_ids, train_subjects))[0], \
                             np.where(np.isin(self.subject_ids, val_subjects))[0]
        return Subset(self, train_ids), Subset(self, val_ids)

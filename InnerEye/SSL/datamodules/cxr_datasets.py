#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import PIL
from PIL import Image

import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset
import pydicom as dicom

class InnerEyeCXRDatasetBase(VisionDataset):
    """
    Base class for dataset with Chest X-ray data. Implements reading of dicom as well as png.
    """
    def __init__(self,
                 data_directory: str,
                 train: bool,
                 seed: int = 1234,
                 transform: Optional[Callable] = None,
                 return_index: bool = False,
                 **kwargs) -> None:
        super().__init__(root=data_directory, transforms=transform)
        self.root = Path(self.root)
        if not self.root.exists():
            logging.error(
                f"The data directory {self.root} does not exist. Make sure to download the data first for the Kaggle "
                f"page")
        self.train = train
        self.return_index = return_index
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self._prepare_dataset()
        dataset_type = "TRAIN" if self.train else "VAL"
        logging.info(f"Number samples - {dataset_type}: {len(self)}")
        if self.targets is not None:
            logging.info(f"Proportion of positive labels - {dataset_type}: {np.mean(self.targets)}")

    def _prepare_dataset(self):
        self.indices = None
        self.filenames = None
        self.targets = None
        raise  NotImplementedError("_prepare_dataset needs to be implemented by the child classes.")

    def __getitem__(self, index: int) -> Union[Tuple[PIL.Image.Image, int], Tuple[int, PIL.Image.Image, int]]:
        """
        :param index: The index of the sample to be fetched
        :return: The image and (fake) label tensors
        """
        filename = self.filenames[index]
        target = self.targets[index] if self.targets is not None else 0
        if str(filename).endswith("dcm"):
            scan_image = self.read_dicom(filename)
            scan_image = (scan_image - scan_image.min()) * 255. / (scan_image.max() - scan_image.min())
            scan_image = Image.fromarray(scan_image).convert("L")
        else:
            scan_image = Image.open(filename).convert("L")
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        if self.return_index:
            return index, scan_image, target
        return scan_image, target

    def __len__(self) -> int:
        """
        :return: The size of the dataset
        """
        return len(self.indices)

    @property
    def num_classes(self) -> int:
        raise NotImplementedError

    #todo switch to proper innereye function when it is merged
    @staticmethod
    def read_dicom(filepath) -> np.array:
        ds = dicom.dcmread(filepath)
        f = ds.pixel_array
        if ds.PhotometricInterpretation == "MONOCHROME1":
            f = np.invert(f)
        return f


class NIH(InnerEyeCXRDatasetBase):
    """
    Dataset class to load the NIH Chest-Xray dataset. Use the full data for training and validation (including
    the official test set).
    """
    def _prepare_dataset(self):
        self.dataset_dataframe = pd.read_csv(self.root / "Data_Entry_2017.csv")
        self.indices = np.arange(len(self.dataset_dataframe))
        self.subject_ids = self.dataset_dataframe["Image Index"].values
        self.filenames = [self.root / f"{subject_id}" for subject_id in self.subject_ids]
        self.targets = None
        # TO EXCLUDE TEST SET FROM TRAINING/VAL DATA
        # train_ids = pd.read_csv(self.root / "train_val_list.txt", header=None).values.reshape(-1)
        # subjects_ids = self.dataset_dataframe["Image Index"].values
        # is_train_val_ids = self.dataset_dataframe["Image Index"].isin(train_ids).values
        # self.indices = np.where(is_train_val_ids)[0] if self.train else np.where(~is_train_val_ids)[0]
        # self.indices = self.random_state.permutation(self.indices)


class RSNAKaggleCXR(InnerEyeCXRDatasetBase):
    """
    Dataset class to load the RSNA Chest-Xray training dataset. Use all the data for train and val. No test data.
    """
    def _prepare_dataset(self):
        self.dataset_dataframe = pd.read_csv(self.root / "dataset.csv")
        self.targets = self.dataset_dataframe.label.values.astype(np.int64)
        self.subject_ids = self.dataset_dataframe.subject.values
        self.indices = np.arange(len(self.dataset_dataframe))
        self.filenames = [self.root / f"{subject_id}.dcm" for subject_id in self.subject_ids]

    @property
    def num_classes(self) -> int:
        return 2
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Dict, Union

import PIL
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

NIH_TOTAL_SIZE = 112120


class NIHCXR(Dataset):
    def __init__(self,
                 data_directory: str,
                 use_training_split: bool,
                 seed: int = 1234,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 num_samples: int = None,
                 return_index: bool = True) -> None:
        """
        Class for the full NIH ChestXray Dataset (112k images)

        :param data_directory: the directory containing all training images from the dataset as well as the
        Data_Entry_2017.csv file containing the dataset labels.
        :param use_training_split: whether to return the training or the test split of the dataset.
        :param seed: random seed to use for dataset creation
        :param shuffle: whether to shuffle the dataset prior to spliting between validation and training
        :param transform: a preprocessing function that takes a PIL image as input and returns a tensor
        :param num_samples: number of the samples to return (has to been smaller than the dataset split)
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            logging.error(
                f"The data directory {self.data_directory} does not exist. Make sure to download the NIH data "
                f"first.The dataset can on the main page"
                "https://www.kaggle.com/nih-chest-xrays/data. Make sure all images are placed directly under the "
                "data_directory folder. Make sure you downloaded the Data_Entry_2017.csv file to this directory as"
                "well.")

        self.train = use_training_split
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset_dataframe = pd.read_csv(self.data_directory / "Data_Entry_2017.csv")
        self.dataset_dataframe["pneumonia_like"] = self.dataset_dataframe["Finding Labels"].apply(
            lambda x: x.split("|")).apply(lambda x: "pneumonia" in x.lower()
                                                    or "infiltration" in x.lower()
                                                    or "consolidation" in x.lower())
        self.transforms = transform

        orig_labels = self.dataset_dataframe.pneumonia_like.values.astype(np.int64)
        subjects_ids = self.dataset_dataframe["Image Index"].values
        is_train_ids = self.dataset_dataframe["train"].values
        self.num_classes = 2
        self.indices = np.where(is_train_ids)[0] if use_training_split else np.where(~is_train_ids)[0]
        self.indices = self.random_state.permutation(self.indices) \
            if shuffle else self.indices
        # ------------- Select subset of current split ------------- #
        if num_samples is not None:
            assert 0 < num_samples <= len(self.indices)
            self.indices = self.indices[:num_samples]

        self.subject_ids = subjects_ids[self.indices]
        self.orig_labels = orig_labels[self.indices].reshape(-1)
        self.targets = self.orig_labels

        # Identify case ids for ambiguous and clear label noise cases
        self.ambiguity_metric_args: Dict = dict()

        dataset_type = "TRAIN" if use_training_split else "VAL"
        logging.info(f"Proportion of positive labels - {dataset_type}: {np.mean(self.targets)}")
        logging.info(f"Number samples - {dataset_type}: {self.targets.shape[0]}")
        self.return_index = return_index

    def __getitem__(self, index: int) -> Union[Tuple[int, PIL.Image.Image, int], Tuple[PIL.Image.Image, int]]:
        """

        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        subject_id = self.subject_ids[index]
        filename = self.data_directory / f"{subject_id}"
        target = self.targets[index]
        scan_image = Image.open(filename).convert("L")
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        if self.return_index:
            return index, scan_image, int(target)
        return scan_image, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.indices)

    def get_label_names(self) -> List[str]:
        return ["NotPneunomiaLike", "PneunomiaLike"]

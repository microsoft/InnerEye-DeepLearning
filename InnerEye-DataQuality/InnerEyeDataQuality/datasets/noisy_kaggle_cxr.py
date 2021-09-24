#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import PIL
import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from torch.utils.data import Dataset

from InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from InnerEyeDataQuality.selection.simulation_statistics import SimulationStats
from InnerEyeDataQuality.utils.generic import convert_labels_to_one_hot
from InnerEyeDataQuality.evaluation.metrics import compute_label_entropy


class NoisyKaggleSubsetCXR(Dataset):
    def __init__(self, data_directory: str,
                 use_training_split: bool,
                 consolidation_noise_rate: float,
                 train_fraction: float = 0.5,
                 seed: int = 1234,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 num_samples: Optional[int] = None,
                 use_noisy_fixed_labels: bool = True) -> None:
        """
        Class for the noisy Kaggle RSNA Pneumonia Detection Dataset. This dataset uses the kaggle dataset with noisy
        labels
        as the original labels from RSNA and the clean labels are the Kaggle labels.

        :param data_directory: the directory containing all training images from the Challenge (stage 1) as well as the
        dataset.csv containing the kaggle and the original labels.
        :param use_training_split: whether to return the training or the validation split of the dataset.
        :param train_fraction: the proportion of samples to use for training
        :param seed: random seed to use for dataset creation
        :param shuffle: whether to shuffle the dataset prior to spliting between validation and training
        :param transform: a preprocessing function that takes a PIL image as input and returns a tensor
        :param num_samples: number of the samples to return (has to been smaller than the dataset split)
        :param use_noisy_fixed_labels: if True use the original labels as the initial labels else use the clean labels.
        :param consolidation_noise_rate: proportion of noisy samples among consolidation/infiltration NIH category.
        """
        dataset_type = "TRAIN" if use_training_split else "VAL"
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            raise RuntimeError(
                f"The data directory {self.data_directory} does not exist. Make sure to download to Kaggle data "
                f"first.The kaggle dataset can "
                "be acceded via the Kaggle CLI kaggle competitions download -c rsna-pneumonia-detection-challenge or "
                "on the main page of the challenge "
                "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images")

        path_to_noisy_csv = Path(__file__).parent / "noisy_chestxray_dataset.csv"
        if not path_to_noisy_csv.exists():
            raise RuntimeError(f"The noisy dataset csv can not be found in {path_to_noisy_csv}, make sure to run "
            "create_noisy_chestxray_dataset.py first. See readme for more detailed instructions on the pre-requisite"
            " for running the noisy Chest Xray benchmark.")

        self.train = use_training_split
        self.train_fraction = train_fraction
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.dataset_dataframe = pd.read_csv(str(Path(__file__).parent / "noisy_chestxray_dataset.csv"))
        self.transforms = transform

        self.dataset_dataframe["initial_label"] = self.dataset_dataframe.binary_nih_initial_label

        # Random uniform noise among consolidation
        pneumonia_like_subj = self.dataset_dataframe.loc[self.dataset_dataframe.nih_pneumonia_like, "subject"].values
        selected = self.random_state.choice(pneumonia_like_subj,
                                            replace=False,
                                            size=int(len(pneumonia_like_subj) * consolidation_noise_rate))
        self.dataset_dataframe.loc[self.dataset_dataframe.subject.isin(selected), "initial_label"] = \
            ~self.dataset_dataframe.loc[self.dataset_dataframe.subject.isin(selected), "initial_label"]

        self.dataset_dataframe["is_noisy"] = self.dataset_dataframe.label_kaggle != self.dataset_dataframe.initial_label

        initial_labels = self.dataset_dataframe.initial_label.values.astype(np.int64).reshape(-1, 1)
        kaggle_labels = self.dataset_dataframe.label_kaggle.values.astype(np.int64).reshape(-1, 1)
        subjects_ids = self.dataset_dataframe.subject.values
        is_ambiguous = self.dataset_dataframe.ambiguous.values
        orig_label = self.dataset_dataframe.orig_label.values
        nih_category = self.dataset_dataframe.nih_category.values

        # Convert clean labels to one-hot to populate label counts
        # i.e for easy cases assume the true distribution is 100% ground truth
        kaggle_label_counts = convert_labels_to_one_hot(kaggle_labels, n_classes=2)
        # For ambiguous cases: [0, 1] -> [1, 2] and [1, 0] -> [2, 1]
        kaggle_label_counts[is_ambiguous, :] = kaggle_label_counts[is_ambiguous, :] * 2 + 1
        _, self.num_classes = kaggle_label_counts.shape
        assert self.num_classes == 2

        # ------------- Split the data into training and validation sets ------------- #
        self.num_datapoints = len(self.dataset_dataframe)
        all_indices = np.arange(self.num_datapoints)
        num_samples_set1 = int(self.num_datapoints * self.train_fraction)
        all_indices = self.random_state.permutation(all_indices) \
            if shuffle else all_indices
        train_indices = all_indices[:num_samples_set1]
        val_indices = all_indices[num_samples_set1:]
        self.indices = train_indices if use_training_split else val_indices

        # ------------- Select subset of current split ------------- #
        # If n_samples is set to restrict dataset i.e. for data_curation
        num_samples = self.num_datapoints if num_samples is None else num_samples
        if num_samples < self.num_datapoints:
            assert 0 < num_samples <= len(self.indices)
        self.indices = self.indices[:num_samples]

        # ------------ Finalize dataset --------------- #
        self.subject_ids = subjects_ids[self.indices]

        # Label distribution is constructed from the true labels
        self.label_counts = kaggle_label_counts[self.indices]
        self.label_distribution = LabelDistribution(seed, self.label_counts)

        self.initial_labels = initial_labels[self.indices].reshape(-1)
        self.kaggle_labels = kaggle_labels[self.indices].reshape(-1)
        self.targets = self.initial_labels if use_noisy_fixed_labels else self.kaggle_labels
        self.orig_labels = orig_label[self.indices]
        self.is_ambiguous = is_ambiguous[self.indices]
        self.nih_category = nih_category[self.indices]

        # Identify case ids for ambiguous and clear label noise cases
        label_stats = SimulationStats(name="NoisyChestXray", true_label_counts=self.label_counts,
                                      initial_labels=convert_labels_to_one_hot(self.targets, self.num_classes))
        self.clear_mislabeled_cases = label_stats.mislabelled_not_ambiguous_sample_ids[0]
        self.ambiguous_mislabelled_cases = label_stats.mislabelled_ambiguous_sample_ids[0]
        self.true_label_entropy = compute_label_entropy(label_counts=self.label_counts)
        self.ambiguity_metric_args = {"ambiguous_mislabelled_ids": self.ambiguous_mislabelled_cases,
                                      "clear_mislabelled_ids": self.clear_mislabeled_cases,
                                      "true_label_entropy": self.true_label_entropy}
        self.num_samples = self.targets.shape[0]
        logging.info(self.num_samples)
        logging.info(len(self.targets))
        logging.info(len(self.indices))
        logging.info(f"Proportion of positive clean labels - {dataset_type}: {np.mean(self.kaggle_labels)}")
        logging.info(f"Proportion of positive noisy labels - {dataset_type}: {np.mean(self.targets)}")
        logging.info(
            f"Total noise rate on the {dataset_type} dataset: {np.mean(self.kaggle_labels != self.targets)} \n")
        selected_df = self.dataset_dataframe.loc[self.dataset_dataframe.subject.isin(self.subject_ids)]
        noisy_df = selected_df.loc[selected_df.is_noisy]
        noisy_df["nih_noise"] = ~noisy_df.nih_pneumonia_like
        logging.info(f"\n{pd.crosstab(noisy_df.nih_noise, noisy_df.ambiguous).to_string()}")
        # self.weight = np.mean(self.kaggle_labels)
        # logging.info(f"Weight negative {self.weight:.2f} - weight positive {(1 - self.weight):.2f}")
        self.png_files = (self.data_directory / f"{self.subject_ids[0]}.png").exists()

    def __getitem__(self, index: int) -> Tuple[int, PIL.Image.Image, int]:
        """

        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        subject_id = self.subject_ids[index]
        target = self.targets[index]
        if self.png_files:
            filename = self.data_directory / f"{subject_id}.png"
            scan_image = Image.open(filename)
        else:
            filename = self.data_directory / f"{subject_id}.dcm"
            scan_image = dicom.dcmread(filename).pixel_array
            scan_image = Image.fromarray(scan_image)
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        if scan_image.shape == 2:
            scan_image = scan_image.unsqueeze(dim=0)
        return index, scan_image, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.subject_ids)

    def get_label_names(self) -> List[str]:
        return ["Normal", "Opacity"]

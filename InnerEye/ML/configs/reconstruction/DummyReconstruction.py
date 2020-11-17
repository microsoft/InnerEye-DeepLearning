#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any

import h5py
import pandas as pd
import param
import torch

from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.deep_learning_config import ModelCategory
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.reconstruction_config import ReconstructionModelBase
from InnerEye.ML.dataset import fastmri_dataset
from InnerEye.Reconstruction.recon_utils import fft 


class SimpleMriNet(torch.nn.Module):
    def __init__(self):
        super(SimpleMriNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        x = fft.ifft(x, dim=(-2,-1))
        x = torch.sqrt(torch.abs(torch.sum(x**2, dim=1)))
        x = torch.nn.functional.relu(self.conv1(x.unsqueeze(1))).squeeze(1)
        return x

class DummyReconstruction(ReconstructionModelBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

        # TODO: this should maybe be something else, but we have to set something for now.
        self._model_category = ModelCategory.Regression

    def validate(self) -> None:
        """
        Overloading validate to enable dataset.csv creation
        """
        super().validate()
        self.generate_dataset_csv_if_missing()

    def read_dataset_into_dataframe_and_pre_process(self) -> None:
        """
        Loads a dataset from the dataset.csv file, and stores it in the present object.
        """
        assert self.local_dataset is not None  # for mypy
        self.dataset_data_frame = pd.read_csv(self.local_dataset / DATASET_CSV_FILE_NAME,
                                              dtype={'SubjectId': str,
                                                     'FilePath': str,
                                                     'SliceIndex': int,
                                                     'Coils': int,
                                                     'SizeKx': int,
                                                     'SizeKy': int,
                                                     'SizeX': int,
                                                     'SizeY': int},
                                              low_memory=False)
        self.pre_process_dataset_dataframe()
    
    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
            subject_column='SubjectId'
        )

    def create_and_set_torch_datasets(self, for_training: bool = True, for_inference: bool = True) -> None:
        """
        Creates torch datasets for all model execution modes, and stores them in the object.
        """

        dataset_splits = self.get_dataset_splits()
        if for_training:
            self._datasets_for_training = {
                ModelExecutionMode.TRAIN: fastmri_dataset.FastMriDataset(self, dataset_splits.train),  # type: ignore
                ModelExecutionMode.VAL: fastmri_dataset.FastMriDataset(self, dataset_splits.val),  # type: ignore
            }
        if for_inference:
            self._datasets_for_inference = {
                mode: fastmri_dataset.FastMriDataset(
                    self,
                    dataset_splits[mode])  # type: ignore
                for mode in ModelExecutionMode if len(dataset_splits[mode]) > 0
            }

    def create_model(self) -> Any:
        """
        Creates a dummy torch model from the provided arguments and returns a torch.nn.Module object.
        """
        return SimpleMriNet()

    def generate_dataset_csv_if_missing(self) -> None:
        if self.local_dataset:
            dataset_csv = self.local_dataset / DATASET_CSV_FILE_NAME
            if not dataset_csv.is_file():
                df = pd.DataFrame(columns = ['SubjectId', 'FilePath', 'SliceIndex', 'Coils', 'SizeKx', 'SizeKy', 'SizeX', 'SizeY'])
                for f in self.local_dataset.glob("*.h5"):
                    with h5py.File(f) as d:
                        kspace_shape = d[fastmri_dataset.KSPACE_NAME].shape
                        recon_shape = d[fastmri_dataset.RECONSTRUCTION_NAME].shape
                        for s in range(kspace_shape[0]):
                            print('Appending slice')
                            df = df.append({'SubjectId': f.name + str(s),
                                            'FilePath': str(f),
                                            'SliceIndex': s,
                                            'Coils': kspace_shape[1],
                                            'SizeKx': kspace_shape[2],
                                            'SizeKy': kspace_shape[3],
                                            'SizeX': recon_shape[1],
                                            'SizeY': recon_shape[2]}, ignore_index=True)
                df.to_csv(dataset_csv)    

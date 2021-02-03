#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pandas as pd

from InnerEye.ML.configs.segmentation.BasicModel2Epochs import BasicModel2Epochs
from InnerEye.ML.utils.split_dataset import DatasetSplits


class BasicModel2EpochsMoreData(BasicModel2Epochs):
    """
    A clone of the basic PR build model, that has more training data, to avoid PyTorch throwing failures
    because each rank does not have enough data to train on.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_subject_ids(
            df=dataset_df,
            train_ids=['0', '1', '2', '3'],
            test_ids=['4', '5', '6', '7'],
            val_ids=['8', '9']
        )

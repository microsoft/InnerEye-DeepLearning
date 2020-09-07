#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable

import pandas as pd
import torch

from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path
from Tests.ML.models.architectures.DummyScalarModel import DummyScalarModel


class ClassificationModelForTesting(ScalarModelBase):
    def __init__(self, conv_in_3d: bool = True, mean_teacher_model: bool = False) -> None:
        num_epochs = 4
        super().__init__(
            local_dataset=full_ml_test_data_path("classification_data"),
            image_channels=["image"],
            image_file_column="path",
            label_channels=["label"],
            label_value_column="value",
            non_image_feature_channels={},
            numerical_columns=[],
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            test_start_epoch=num_epochs,
            subject_column="subjectID",
            compute_mean_teacher_model=mean_teacher_model
        )
        self.expected_image_size_zyx = (4, 5, 7)
        self.conv_in_3d = conv_in_3d

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
            random_seed=1,
            subject_column=self.subject_column
        )

    def create_model(self) -> Any:
        return DummyScalarModel(self.expected_image_size_zyx)

    def get_post_loss_logits_normalization_function(self) -> Callable:
        return torch.nn.Sigmoid()

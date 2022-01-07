#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable

import pandas as pd
import torch

from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.ML.models.architectures.DummyScalarModel2D import DummyScalarModel2D
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path


class ClassificationModelForTesting2D(ScalarModelBase):
    def __init__(self, conv_in_3d: bool = True, mean_teacher_model: bool = False) -> None:
        num_epochs = 4
        mean_teacher_alpha = 0.99 if mean_teacher_model else None
        super().__init__(
            local_dataset=full_ml_test_data_path("classification_data_2d"),
            image_channels=["image"],
            image_file_column="path",
            label_channels=["label"],
            label_value_column="value",
            non_image_feature_channels={},
            numerical_columns=[],
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            subject_column="subjectID",
            mean_teacher_alpha=mean_teacher_alpha
        )
        self.expected_image_size_zyx = (5, 7)
        self.conv_in_3d = conv_in_3d
        self.pl_deterministic = True

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
        return DummyScalarModel2D(self.expected_image_size_zyx)

    def get_post_loss_logits_normalization_function(self) -> Callable:
        return torch.nn.Sigmoid()

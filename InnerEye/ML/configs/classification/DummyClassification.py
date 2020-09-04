#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd

from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path


class DummyClassification(ScalarModelBase):
    "A config file for dummy image classification model for debugging purposes" 

    def __init__(self) -> None:
        num_epochs = 4
        super().__init__(
            local_dataset=full_ml_test_data_path("classification_data"),
            image_channels=["image"],
            image_file_column="path",
            label_channels=["label"],
            label_value_column="value",
            non_image_feature_channels=[],
            numerical_columns=[],
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            test_start_epoch=num_epochs,
            use_mixed_precision=True,
            subject_column="subjectID",
            conv_in_3d=True
        )
        self.expected_image_size_zyx = (4, 5, 7)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
            subject_column=self.subject_column
        )

    def create_model(self) -> Any:
        # Use a local import so that we don't need to import pytorch when creating configs in the runner
        from Tests.ML.models.architectures.DummyScalarModel import DummyScalarModel
        return DummyScalarModel(self.expected_image_size_zyx)

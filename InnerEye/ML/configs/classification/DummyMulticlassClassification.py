#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import pandas as pd

from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path


class DummyMulticlassClassification(ScalarModelBase):
    "A config file for dummy image classification model for debugging purposes"

    def __init__(self) -> None:
        num_epochs = 4
        super().__init__(
            local_dataset=full_ml_test_data_path("classification_data_multiclass"),
            image_channels=["blue"],
            image_file_column="path",
            label_channels=["blue"],
            class_names=["class0", "class1", "class2", "class3", "class4"],
            # labels_exclusive=False,
            label_value_column="label",
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            use_mixed_precision=True,
            subject_column="ID",
            image_size=(1, 5, 7)
        )
        self.conv_in_3d = True
        self.expected_image_size_zyx = (1, 5, 7)
        # Trying to run DDP from the test suite hangs, hence restrict to single GPU.
        self.max_num_gpus = 1

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
        return DummyScalarModel(self.expected_image_size_zyx, num_classes=len(self.class_names))

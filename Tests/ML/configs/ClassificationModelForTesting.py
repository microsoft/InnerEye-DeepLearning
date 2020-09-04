#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, List

import pandas as pd
import torch

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.layers.identity import Identity
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path


class DummyScalarModel(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self, expected_image_size_zyx: TupleInt3,
                 activation: torch.nn.Module = Identity()) -> None:
        super().__init__()
        self.expected_image_size_zyx = expected_image_size_zyx
        self._layers = torch.nn.ModuleList()
        fc = torch.nn.Conv3d(1, 1, kernel_size=3)
        torch.nn.init.normal_(fc.weight, 0, 0.01)
        with torch.no_grad():
            fc_out = fc(torch.zeros((1, 1) + self.expected_image_size_zyx))
            self.feature_size = fc_out.view(-1).shape[0]
        self._layers.append(fc)
        self.fc = torch.nn.Linear(self.feature_size, 1)
        self.activation = activation
        self.last_encoder_layer: List[str] = ["_layers", "0"]
        self.conv_in_3d = False

    def get_last_encoder_layer_names(self) -> List[str]:
        return self.last_encoder_layer

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into images
        :param item: ClassificationItem
        :return: Tensor
        """
        return [item.images]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if x.shape[-3:] != self.expected_image_size_zyx:
            raise ValueError(f"Expected a tensor with trailing size {self.expected_image_size_zyx}, but got "
                             f"{x.shape}")

        for layer in self._layers.__iter__():
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.activation(x)


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

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import pandas as pd

from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits

# Change this string to the name of your dataset on Azure blob storage.
AZURE_DATASET_ID = "glaucoma_public_dataset"


class GlaucomaPublic(ScalarModelBase):
    """
    A config file implementing a classification model to diagnose patients with glaucoma pathology
    by analysing input retinal OCT scans. Reference:
    Maetschke, Stefan, et al. "A feature agnostic approach for glaucoma detection in OCT volumes." PloS one 14.7 (2019)
    """

    def __init__(self, **kwargs: Any) -> None:
        num_epochs = 50
        super().__init__(
            azure_dataset_id=AZURE_DATASET_ID,
            image_channels=["image"],
            image_file_column="filePath",
            label_channels=["image"],
            label_value_column="label",
            non_image_feature_channels=[],
            numerical_columns=[],
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            epochs_to_test=[num_epochs],
            use_mixed_precision=True,
            train_batch_size=64,  # Batch size of 64 uses about 7GB of GPU memory
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.8,
            proportion_test=0.1,
            proportion_val=0.1,
        )

    def create_model(self) -> Any:
        # Use a local import so that we don't need to import pytorch when creating configs in the runner
        import torch
        from InnerEye.ML.config import PaddingMode
        from InnerEye.ML.dataset.scalar_sample import ScalarItem
        from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
        from InnerEye.ML.models.layers.basic import BasicLayer

        class ModelFromPaper(DeviceAwareModule[ScalarItem, torch.Tensor]):
            def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
                return [item.images]

            def __init__(self) -> None:
                super().__init__()

                num_classes = 1
                num_conv_layers = 5
                kernel_size = [7, 5, 3, 3, 3]
                channels = [1, 32, 32, 32, 32, 32]
                stride = [2, 1, 1, 1, 1]
                padding = [PaddingMode.Zero] * num_conv_layers

                _conv_layers = []

                # Convolution Layers
                for ii in range(num_conv_layers):
                    _conv_layers.append(
                        BasicLayer(channels=(channels[ii], channels[ii + 1]),
                                   kernel_size=kernel_size[ii],
                                   stride=stride[ii],
                                   padding=padding[ii],
                                   activation=torch.nn.ReLU))

                # Pooling and dense layers
                self.encoder = torch.nn.Sequential(*_conv_layers)
                self.aggregation_layer = torch.nn.functional.avg_pool3d
                self.dense_layer = torch.nn.Linear(in_features=channels[-1],
                                                   out_features=num_classes,
                                                   bias=True)
                self.conv_in_3d = True
                self.last_encoder_layer = ["encoder", "4", "activation"]

            def get_last_encoder_layer_names(self) -> List[str]:
                return self.last_encoder_layer

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
                x = self.encoder(x)
                x = self.aggregation_layer(input=x, kernel_size=x.shape[2:])
                x = self.dense_layer(x.view(-1, x.shape[1]))
                return x

        return ModelFromPaper()

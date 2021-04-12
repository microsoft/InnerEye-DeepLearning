#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

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
        from InnerEye.ML.utils.model_util import build_glaucoma_net
        return build_glaucoma_net(self)

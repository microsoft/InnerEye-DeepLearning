#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.model_training import model_train
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import create_mlp
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits

from Tests.ML.util import get_default_checkpoint_handler


class NonImageEncoder(ScalarModelBase):
    def __init__(self, hidden_layer_num_feature_channels: Optional[int] = None, **kwargs: Any) -> None:
        num_epochs = 3
        super().__init__(
            # label_channels="16",
            label_value_column="label",
            label_channels=["12"],
            non_image_feature_channels=["0", "8", "12"],
            numerical_columns=["NUM1", "NUM2"],
            categorical_columns=["CAT1", "CAT2"],
            local_dataset=Path(),
            channel_column='channel',
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            test_start_epoch=num_epochs,
            train_batch_size=2,
            l_rate=1e-1,
            **kwargs
        )
        self.hidden_layer_num_feature_channels = hidden_layer_num_feature_channels

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
        )

    def create_model(self) -> Any:
        return create_mlp(input_num_feature_channels=self.get_total_number_of_non_imaging_features(),
                          dropout=0.5,
                          hidden_layer_num_feature_channels=self.hidden_layer_num_feature_channels)


@pytest.mark.skipif(common_util.is_windows(), reason="Has issue on Windows build")
@pytest.mark.parametrize("hidden_layer_num_feature_channels", [None, 2])
def test_non_image_encoder(test_output_dirs: OutputFolderForTests,
                           hidden_layer_num_feature_channels: Optional[int]) -> None:
    """
    Test if we can build a simple MLP model that only feeds off non-image features.
    """
    dataset_folder = Path(test_output_dirs.make_sub_dir("dataset"))
    dataset_contents = _get_fake_dataset_contents()
    (dataset_folder / DATASET_CSV_FILE_NAME).write_text(dataset_contents)
    config = NonImageEncoder(should_validate=False, hidden_layer_num_feature_channels=hidden_layer_num_feature_channels)
    config.local_dataset = dataset_folder
    config.max_batch_grad_cam = 1
    config.validate()
    # run model training
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=Path(test_output_dirs.root_dir))
    model_train(config, checkpoint_handler=checkpoint_handler)
    # run model inference
    MLRunner(config).model_inference_train_and_test(checkpoint_handler=checkpoint_handler)
    assert config.get_total_number_of_non_imaging_features() == 18


def _get_fake_dataset_contents() -> str:
    """
    Build a fake dataset file and write it to a temporary folder.
    The dataset has "measurements" for 3 different channels 0, 8, and 12, with columns for two
    numerical features NUM1, NUM2 and two categorical features CAT1, CAT".
    Labels are attached to channel 12 only.
    """
    return """subject,channel,label,NUM1,NUM2,CAT1,CAT2
    S1,0,,1.0,11.0,False,catA
    S1,8,,1.8,11.8,True,catA
    S1,12,False,1.12,11.12,True,catA
    S2,0,,2.0,22.0,True,catB
    S2,8,,2.8,22.8,True,catB
    S2,12,True,2.12,22.12,False,catB
    S3,0,,3.0,33.0,False,catB
    S3,8,,3.8,33.8,False,catB
    S3,12,True,3.12,33.12,False,catB
    S4,0,,1.0,11.0,False,catA
    S4,8,,1.8,11.8,False,catA
    S4,12,False,1.12,11.12,False,catA
    S5,0,,2.0,22.0,True,catB
    S5,8,,2.8,22.8,True,catB
    S5,12,True,2.12,22.12,True,catB
    S6,0,,3.0,33.0,False,catA
    S6,8,,3.8,33.8,False,catA
    S6,12,True,3.12,33.12,False,catA
    """

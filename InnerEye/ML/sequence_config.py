#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Optional

import param

from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import TemperatureScalingConfig
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils.split_dataset import DatasetSplits

SEQUENCE_POSITION_HUE_NAME_PREFIX = "Seq_pos"


class SequenceModelBase(ScalarModelBase):
    sequence_column: Optional[str] = \
        param.String(allow_None=True, default=None,
                     doc="If provided, create a sequence dataset, ordering by the column given here. The value in that "
                         "column is expected to be numeric, starting at 0.")

    min_sequence_position_value: int = \
        param.Integer(default=0, bounds=(0, None),
                      doc="When creating a sequence dataset, restrict it to items with a sequence index at min this "
                          "value. For example, if sequence_min_index==2, only items with sequence positions >= 2 will "
                          "be retained, and sequences that don't have all positions up to (including) 2 will be "
                          "discarded.")

    max_sequence_position_value: Optional[int] = \
        param.Integer(default=None, allow_None=True, bounds=(0, None),
                      doc="When creating a sequence dataset, restrict it to items with a sequence index "
                          "at most this value. For example, if sequence_max_index==3, only items with sequence "
                          "positions 0, 1, 2, and 3 will be retained.")

    sequence_target_positions: List[int] = \
        param.List(class_=int,
                   doc="Stores the sequence positions for which the model should make predictions. Sequence positions "
                       "are given by the value in the sequence_column. For example, if a sequence consists of items "
                       "with positions [2, 3, 4, 5], and sequence_target_position==[2,5], the model would be evaluated "
                       "on the first and last sequence elements.")

    temperature_scaling_config: Optional[TemperatureScalingConfig] = param.ClassSelector(
        class_=TemperatureScalingConfig,
        allow_None=True,
        default=None,
        doc="If a config is provided then it will be used to learn a temperature scaling parameter using the "
            "validation set to calibrate the model logits see: https://arxiv.org/abs/1706.04599 for each "
            "epoch that requires a checkpoint to be saved. Turned off by default.")

    def __init__(self, **params: Any):
        super().__init__(**params)
        if len(self.sequence_target_positions) == 0:
            raise ValueError("sequence_target_positions must not be empty")
        if self.temperature_scaling_config:
            logging.info(f"Temperature scaling will be performed on the "
                         f"validation set using the config: {self.temperature_scaling_config}")

    def get_total_number_of_validation_epochs(self) -> int:
        num_val_epochs = super().get_total_number_of_validation_epochs()
        if self.temperature_scaling_config:
            # as temperature scaling will be performed for each checkpoint epoch
            # make sure this is accounted for in the allowed repeats of the validation data loader
            num_val_epochs += self.get_total_number_of_save_epochs()
        return num_val_epochs

    def get_target_indices(self) -> List[int]:
        """
        Computes the zero based array indices inside of a sequence of items
        for which the model should make predictions.
        """
        return [pos - self.min_sequence_position_value for pos in self.sequence_target_positions]

    def get_total_number_of_numerical_non_imaging_features(self) -> int:
        return len(self.numerical_columns)

    def get_total_number_of_categorical_non_imaging_features(self) -> int:
        if self.categorical_feature_encoder:
            return sum([self.categorical_feature_encoder.get_feature_length(col) for col in self.categorical_columns])
        else:
            return 0

    @property
    def is_non_imaging_model(self) -> bool:
        """
        Returns whether the model uses non image features only
        """
        return self.image_file_column is None

    def create_torch_datasets(self, dataset_splits: DatasetSplits) -> Dict[ModelExecutionMode, Any]:
        from InnerEye.ML.dataset.sequence_dataset import SequenceDataset
        sample_transforms = self.get_image_sample_transforms()
        train = SequenceDataset(self, dataset_splits.train, name="training",
                                sample_transforms=sample_transforms.train)  # type: ignore
        val = SequenceDataset(self, dataset_splits.val, feature_statistics=train.feature_statistics, name="validation",
                              sample_transforms=sample_transforms.val)  # type: ignore
        test = SequenceDataset(self, dataset_splits.test, feature_statistics=train.feature_statistics, name="test",
                               sample_transforms=sample_transforms.test)  # type: ignore

        return {
            ModelExecutionMode.TRAIN: train,
            ModelExecutionMode.VAL: val,
            ModelExecutionMode.TEST: test
        }

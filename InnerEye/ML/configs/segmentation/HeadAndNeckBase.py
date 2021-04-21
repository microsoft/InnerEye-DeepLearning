#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, List, Optional

import pandas as pd

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import MixtureLossComponent, PhotometricNormalizationMethod, SegmentationLoss, \
    SegmentationModelBase, SliceExclusionRule, SummedProbabilityRule, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list
from InnerEye.ML.utils.split_dataset import DatasetSplits

RANDOM_COLOUR_GENERATOR = random.Random(0)


# This configuration needs to be supplied with a value for azure_dataset_id that refers to your
# dataset. You may also supply a value for num_structures, feature_channels or any other feature. For example,
# with the appropriate dataset, this would build the model whose results are reported in the InnerEye team's
# paper:
#
# class HeadAndNeckPaper(HeadAndNeckBase):
#
#     def __init__(self):
#         super().__init__(
#             azure_dataset_id="foo_bar_baz",
#             num_structures=10)

class HeadAndNeckBase(SegmentationModelBase):
    """
    Head and Neck radiotherapy image segmentation model.
    """

    def __init__(self,
                 ground_truth_ids: List[str],
                 ground_truth_ids_display_names: Optional[List[str]] = None,
                 colours: Optional[List[TupleInt3]] = None,
                 fill_holes: Optional[List[bool]] = None,
                 roi_interpreted_types: Optional[List[str]] = None,
                 class_weights: Optional[List[float]] = None,
                 slice_exclusion_rules: Optional[List[SliceExclusionRule]] = None,
                 summed_probability_rules: Optional[List[SummedProbabilityRule]] = None,
                 num_feature_channels: Optional[int] = None,
                 **kwargs: Any) -> None:
        """
        Creates a new instance of the class.
        :param ground_truth_ids: List of ground truth ids.
        :param ground_truth_ids_display_names: Optional list of ground truth id display names. If
        present then must be of the same length as ground_truth_ids.
        :param colours: Optional list of colours. If
        present then must be of the same length as ground_truth_ids.
        :param fill_holes: Optional list of fill hole flags. If
        present then must be of the same length as ground_truth_ids.
        :param roi_interpreted_types: Optional list of roi_interpreted_types. If
        present then must be of the same length as ground_truth_ids.
        :param class_weights: Optional list of class weights. If
        present then must be of the same length as ground_truth_ids + 1.
        :param slice_exclusion_rules: Optional list of SliceExclusionRules.
        :param summed_probability_rules: Optional list of SummedProbabilityRule.
        :param num_feature_channels: Optional number of feature channels.
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        """
        # Number of training epochs
        num_epochs = 120
        num_structures = len(ground_truth_ids)
        colours = colours or generate_random_colours_list(RANDOM_COLOUR_GENERATOR, num_structures)
        fill_holes = fill_holes or [True] * num_structures
        roi_interpreted_types = roi_interpreted_types or ["ORGAN"] * num_structures
        ground_truth_ids_display_names = ground_truth_ids_display_names or [f"zz_{x}" for x in ground_truth_ids]
        # The amount of GPU memory required increases with both the number of structures and the
        # number of feature channels. The following is a sensible default to avoid out-of-memory,
        # but you can override is by passing in another (singleton list) value for feature_channels
        # from a subclass.
        num_feature_channels = num_feature_channels or (32 if num_structures <= 20 else 26)
        bg_weight = 0.02 if len(ground_truth_ids) > 1 else 0.25
        class_weights = class_weights or equally_weighted_classes(ground_truth_ids, background_weight=bg_weight)
        # In case of vertical overlap between brainstem and spinal_cord, we separate them
        # by converting brainstem voxels to cord, as the latter is clinically more sensitive.
        # We do the same to separate SPC and MPC; in this case, the direction of change is unimportant,
        # so we choose SPC-to-MPC arbitrarily.
        slice_exclusion_rules = slice_exclusion_rules or []
        summed_probability_rules = summed_probability_rules or []
        super().__init__(
            should_validate=False,  # we'll validate after kwargs are added
            num_epochs=num_epochs,
            recovery_checkpoint_save_interval=10,
            architecture="UNet3D",
            kernel_size=3,
            train_batch_size=1,
            inference_batch_size=1,
            feature_channels=[num_feature_channels],
            crop_size=(96, 288, 288),
            test_crop_size=(144, 512, 512),
            inference_stride_size=(72, 256, 256),
            image_channels=["ct"],
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=50,
            window=600,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            use_mixed_precision=True,
            use_model_parallel=True,
            monitoring_interval_seconds=0,
            num_dataload_workers=2,
            loss_type=SegmentationLoss.Mixture,
            mixture_loss_components=[MixtureLossComponent(0.5, SegmentationLoss.Focal, 0.2),
                                     MixtureLossComponent(0.5, SegmentationLoss.SoftDice, 0.1)],
            ground_truth_ids=ground_truth_ids,
            ground_truth_ids_display_names=ground_truth_ids_display_names,
            largest_connected_component_foreground_classes=ground_truth_ids,
            colours=colours,
            fill_holes=fill_holes,
            roi_interpreted_types=roi_interpreted_types,
            class_weights=class_weights,
            slice_exclusion_rules=slice_exclusion_rules,
            summed_probability_rules=summed_probability_rules,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(dataset_df, proportion_train=0.8, proportion_val=0.05,
                                              proportion_test=0.15,
                                              random_seed=0)

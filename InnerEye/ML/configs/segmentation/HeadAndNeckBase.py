#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any

import pandas as pd

from InnerEye.ML.config import MixtureLossComponent, PhotometricNormalizationMethod, SegmentationLoss, \
    SegmentationModelBase, SliceExclusionRule, SummedProbabilityRule, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list
from InnerEye.ML.utils.split_dataset import DatasetSplits

# List of structures to segment. The order is important, because different values of num_structures
# in the constructor will select different prefixes of the list.

STRUCTURE_LIST = ["external", "parotid_l", "parotid_r", "smg_l", "smg_r", "spinal_cord", "brainstem",
                  "globe_l", "globe_r", "mandible", "spc_muscle", "mpc_muscle", "cochlea_l", "cochlea_r",
                  "lens_l", "lens_r", "optic_chiasm", "optic_nerve_l", "optic_nerve_r", "pituitary_gland",
                  "lacrimal_gland_l", "lacrimal_gland_r"]
RANDOM_COLOUR_GENERATOR = random.Random(0)
COLOURS = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, len(STRUCTURE_LIST))
FILL_HOLES = [True] * len(STRUCTURE_LIST)


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

    def __init__(self, num_structures: int = 0, **kwargs: Any) -> None:
        """
        :param num_structures: number of structures from STRUCTURE_LIST to predict (default: all structures)
        :param kwargs: other args from subclass
        """
        # Number of training epochs
        num_epochs = 120
        # Number of structures to predict; if positive but less than the length of STRUCTURE_LIST, the relevant prefix
        # of STRUCTURE_LIST will be predicted.
        if num_structures <= 0 or num_structures > len(STRUCTURE_LIST):
            num_structures = len(STRUCTURE_LIST)
        ground_truth_ids = STRUCTURE_LIST[:num_structures]
        colours = COLOURS[:num_structures]
        fill_holes = FILL_HOLES[:num_structures]
        ground_truth_ids_display_names = [f"zz_{x}" for x in ground_truth_ids]
        # The amount of GPU memory required increases with both the number of structures and the
        # number of feature channels. The following is a sensible default to avoid out-of-memory,
        # but you can override is by passing in another (singleton list) value for feature_channels
        # from a subclass.
        num_feature_channels = 32 if num_structures <= 20 else 26
        bg_weight = 0.02 if len(ground_truth_ids) > 1 else 0.25
        # In case of vertical overlap between brainstem and spinal_cord, we separate them
        # by converting brainstem voxels to cord, as the latter is clinically more sensitive.
        # We do the same to separate SPC and MPC; in this case, the direction of change is unimportant,
        # so we choose SPC-to-MPC arbitrarily.
        slice_exclusion_rules = []
        summed_probability_rules = []
        if "brainstem" in ground_truth_ids and "spinal_cord" in ground_truth_ids:
            slice_exclusion_rules.append(SliceExclusionRule("brainstem", "spinal_cord", False))
            if "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("spinal_cord", "brainstem", "external"))
        if "spc_muscle" in ground_truth_ids and "mpc_muscle" in ground_truth_ids:
            slice_exclusion_rules.append(SliceExclusionRule("spc_muscle", "mpc_muscle", False))
            if "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("mpc_muscle", "spc_muscle", "external"))
        if "optic_chiasm" in ground_truth_ids and "pituitary_gland" in ground_truth_ids:
            slice_exclusion_rules.append(SliceExclusionRule("optic_chiasm", "pituitary_gland", True))
            if "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("optic_chiasm", "pituitary_gland", "external"))
        super().__init__(
            should_validate=False,  # we'll validate after kwargs are added
            num_epochs=num_epochs,
            save_start_epoch=num_epochs,
            save_step_epochs=num_epochs,
            architecture="UNet3D",
            kernel_size=3,
            train_batch_size=4,
            inference_batch_size=1,
            feature_channels=[num_feature_channels],
            crop_size=(96, 288, 288),
            test_crop_size=(144, 512, 512),
            inference_stride_size=(72, 256, 256),
            image_channels=["ct"],
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=50,
            window=600,
            start_epoch=0,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            epochs_to_test=[num_epochs],
            use_mixed_precision=True,
            use_model_parallel=True,
            monitoring_interval_seconds=0,
            num_dataload_workers=4,
            loss_type=SegmentationLoss.Mixture,
            mixture_loss_components=[MixtureLossComponent(0.5, SegmentationLoss.Focal, 0.2),
                                     MixtureLossComponent(0.5, SegmentationLoss.SoftDice, 0.1)],
            ground_truth_ids=ground_truth_ids,
            ground_truth_ids_display_names=ground_truth_ids_display_names,
            largest_connected_component_foreground_classes=ground_truth_ids,
            colours=colours,
            fill_holes=fill_holes,
            class_weights=equally_weighted_classes(ground_truth_ids, background_weight=bg_weight),
            slice_exclusion_rules=slice_exclusion_rules,
            summed_probability_rules=summed_probability_rules,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(dataset_df, proportion_train=0.8, proportion_val=0.05,
                                              proportion_test=0.15,
                                              random_seed=0)

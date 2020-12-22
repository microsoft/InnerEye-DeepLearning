#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import random
from typing import Any, Optional

from InnerEye.ML.config import SliceExclusionRule, SummedProbabilityRule, equally_weighted_classes
from InnerEye.ML.configs.segmentation.HeadAndNeckBase import HeadAndNeckBase
from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list

# List of structures to segment. The order is important, because different values of num_structures
# in the constructor will select different prefixes of the list.

STRUCTURE_LIST = ["external", "parotid_l", "parotid_r", "smg_l", "smg_r", "spinal_cord", "brainstem",
                  "globe_l", "globe_r", "mandible", "spc_muscle", "mpc_muscle", "cochlea_l", "cochlea_r",
                  "lens_l", "lens_r", "optic_chiasm", "optic_nerve_l", "optic_nerve_r", "pituitary_gland",
                  "lacrimal_gland_l", "lacrimal_gland_r"]
RANDOM_COLOUR_GENERATOR = random.Random(0)
COLOURS = generate_random_colours_list(RANDOM_COLOUR_GENERATOR, len(STRUCTURE_LIST))


class HeadAndNeckPaper(HeadAndNeckBase):
    """
    Head and Neck model, as used in the paper.
    """

    def __init__(self, num_structures: Optional[int] = None, **kwargs: Any) -> None:
        """
        Creates a new instance of the class.
        :param num_structures: number of structures from STRUCTURE_LIST to predict (default: all structures)
        :param kwargs: Additional arguments that will be passed through to the SegmentationModelBase constructor.
        """
        # Number of structures to predict; if positive but less than the length of STRUCTURE_LIST, the relevant prefix
        # of STRUCTURE_LIST will be predicted.
        if (num_structures is not None) and \
                (num_structures <= 0 or num_structures > len(STRUCTURE_LIST)):
            raise ValueError(f"num structures must be between 0 and {len(STRUCTURE_LIST)}")
        if num_structures is None:
            logging.info(f'Setting num_structures to: {len(STRUCTURE_LIST)}')
            num_structures = len(STRUCTURE_LIST)
        ground_truth_ids = STRUCTURE_LIST[:num_structures]
        if "ground_truth_ids_display_names" in kwargs:
            ground_truth_ids_display_names = kwargs.pop("ground_truth_ids_display_names")
        else:
            logging.info('Using default ground_truth_ids_display_names')
            ground_truth_ids_display_names = [f"zz_{x}" for x in ground_truth_ids]
        if "colours" in kwargs:
            colours = kwargs.pop("colours")
        else:
            logging.info('Using default colours')
            colours = COLOURS[:num_structures]
        if "fill_holes" in kwargs:
            fill_holes = kwargs.pop("fill_holes")
        else:
            logging.info('Using default fill_holes')
            fill_holes = [True] * num_structures
        # The amount of GPU memory required increases with both the number of structures and the
        # number of feature channels. The following is a sensible default to avoid out-of-memory,
        # but you can override is by passing in another (singleton list) value for feature_channels
        # from a subclass.
        if "num_feature_channels" in kwargs:
            num_feature_channels = kwargs.pop("num_feature_channels")
        else:
            logging.info('Using default num_feature_channels')
            num_feature_channels = 32 if num_structures <= 20 else 26
        bg_weight = 0.02 if len(ground_truth_ids) > 1 else 0.25
        if "class_weights" in kwargs:
            class_weights = kwargs.pop("class_weights")
        else:
            logging.info('Using default class_weights')
            class_weights = equally_weighted_classes(ground_truth_ids, background_weight=bg_weight)
        # In case of vertical overlap between brainstem and spinal_cord, we separate them
        # by converting brainstem voxels to cord, as the latter is clinically more sensitive.
        # We do the same to separate SPC and MPC; in this case, the direction of change is unimportant,
        # so we choose SPC-to-MPC arbitrarily.
        if "slice_exclusion_rules" in kwargs:
            slice_exclusion_rules = kwargs.pop("slice_exclusion_rules")
        else:
            logging.info('Using default slice_exclusion_rules')
            slice_exclusion_rules = []
            if "brainstem" in ground_truth_ids and "spinal_cord" in ground_truth_ids:
                slice_exclusion_rules.append(SliceExclusionRule("brainstem", "spinal_cord", False))
            if "spc_muscle" in ground_truth_ids and "mpc_muscle" in ground_truth_ids:
                slice_exclusion_rules.append(SliceExclusionRule("spc_muscle", "mpc_muscle", False))
            if "optic_chiasm" in ground_truth_ids and "pituitary_gland" in ground_truth_ids:
                slice_exclusion_rules.append(SliceExclusionRule("optic_chiasm", "pituitary_gland", True))

        if "summed_probability_rules" in kwargs:
            summed_probability_rules = kwargs.pop("summed_probability_rules")
        else:
            logging.info('Using default summed_probability_rules')
            summed_probability_rules = []
            if "brainstem" in ground_truth_ids and "spinal_cord" in ground_truth_ids and \
                    "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("spinal_cord", "brainstem", "external"))
            if "spc_muscle" in ground_truth_ids and "mpc_muscle" in ground_truth_ids and \
                    "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("mpc_muscle", "spc_muscle", "external"))
            if "optic_chiasm" in ground_truth_ids and "pituitary_gland" in ground_truth_ids and \
                    "external" in ground_truth_ids:
                summed_probability_rules.append(SummedProbabilityRule("optic_chiasm", "pituitary_gland", "external"))
        super().__init__(
            ground_truth_ids=ground_truth_ids,
            ground_truth_ids_display_names=ground_truth_ids_display_names,
            colours=colours,
            fill_holes=fill_holes,
            class_weights=class_weights,
            slice_exclusion_rules=slice_exclusion_rules,
            summed_probability_rules=summed_probability_rules,
            num_feature_channels=num_feature_channels,
            **kwargs)

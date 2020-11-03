#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from math import isclose
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import param
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import HyperDriveConfig
from pandas import DataFrame

from InnerEye.Common.common_util import any_pairwise_larger, any_smaller_or_equal_than, check_is_any_of
from InnerEye.Common.generic_parsing import IntTuple
from InnerEye.Common.type_annotations import TupleFloat2, TupleFloat3, TupleInt3, TupleStringOptionalFloat
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.deep_learning_config import ModelCategory
from InnerEye.ML.model_config_base import ModelConfigBase, ModelTransformsPerExecutionMode
from InnerEye.ML.utils.split_dataset import DatasetSplits

DATASET_ID_FILE = "dataset_id.txt"
GROUND_TRUTH_IDS_FILE = "ground_truth_ids.txt"
IMAGE_CHANNEL_IDS_FILE = "image_channel_ids.txt"
BACKGROUND_CLASS_NAME = "background"
DEFAULT_POSTERIOR_VALUE_RANGE = (0, 1)
EXAMPLE_IMAGES_FOLDER = "example_images"
LARGEST_CC_TYPE = Optional[Sequence[Union[str, TupleStringOptionalFloat]]]


@unique
class PaddingMode(Enum):
    """
    Supported padding modes for numpy and torch image padding.
    """
    #: Zero padding scheme.
    Zero = 'constant'
    #: Pads with the edge values of array.
    Edge = 'edge'
    #: Pads with the linear ramp between end_value and the array edge value.
    LinearRamp = "linear_ramp"
    #: Pads with the maximum value of all or part of the vector along each axis.
    Maximum = "maximum"
    #: Pads with the mean value of all or part of the vector along each axis.
    Mean = "mean"
    #: Pads with the median value of all or part of the vector along each axis.
    Median = "median"
    #: Pads with the minimum value of all or part of the vector along each axis.
    Minimum = "minimum"
    #: Pads with the reflection of the vector mirrored on the first and last values of the vector along each axis.
    Reflect = "reflect"
    #: Pads with the reflection of the vector mirrored along the edge of the array.
    Symmetric = "symmetric"
    #: Pads with the wrap of the vector along the axis.
    #: The first values are used to pad the end and the end values are used to pad the beginning.
    Wrap = "wrap"
    #: No padding is performed
    NoPadding = "no_padding"


@unique
class EnsembleAggregationType(Enum):
    Average = 'Average'


@unique
class PhotometricNormalizationMethod(Enum):
    """
    Contains the valid methods that can be used to perform photometric normalization of a medical image.
    """
    Unchanged = "None"
    SimpleNorm = "Simple Norm"
    MriWindow = "MRI Window"
    CtWindow = "CT Window"
    TrimmedNorm = "Trimmed Norm"


class ModelArchitectureConfig:
    """
    Supported model architecture types
    """
    Basic = 'Basic'
    UNet3D = 'UNet3D'
    UNet2D = 'UNet2D'


@unique
class SegmentationLoss(Enum):
    """
    The types of training loss that are supported for segmentation models.
    Parameters that can be set in the segmentation configs related to loss functions:

    |  SoftDice: :attr:`SegmentationModelBase.loss_class_weight_power`
    |  CrossEntropy: :attr:`SegmentationModelBase.loss_class_weight_power`,
        :attr:`DeepLearningConfig.label_smoothing_eps`
    |  Focal: :attr:`SegmentationModelBase.loss_class_weight_power`,
        :attr:`DeepLearningConfig.label_smoothing_eps`,
        :attr:`SegmentationModelBase.focal_loss_gamma`
    |  Mixture: :attr:`SegmentationModelBase.mixture_loss_components`.
        See :class:`MixtureLossComponent` for component parameters.
    """
    SoftDice = "SoftDice"
    CrossEntropy = "CrossEntropy"
    Focal = "Focal"
    Mixture = "Mixture"


@dataclass
class MixtureLossComponent:
    """
    A member of the value of the mixture_loss_components parameter.

    Parameters for the loss function will be pulled from the model config,
        except :attr:`SegmentationModelBase.loss_class_weight_power` which is ignored.
    """
    weight: float
    loss_type: SegmentationLoss
    #: For weighted loss, power to which to raise the weights per class.
    class_weight_power: float


@dataclass
class SliceExclusionRule:
    """
    Rule mandating that voxels of higher_class must always be in strictly higher slices than those of lower_class
    (slices are along the z-axis). If this is not the case, then if higher_dominates is True, any lower_class voxels in
    a higher or equal slice to any higher_class voxels are converted to higher_class. If higher_dominates
    is False, any higher_class voxels in a lower or equal slice to any lower_class voxels are converted to lower_class.
    """
    higher_class: str
    lower_class: str
    higher_dominates: bool


@dataclass
class SummedProbabilityRule:
    """
    At the boundary between two classes, the predicted class probability for both classes may be low. To avoid these
    voxels being categorized as external voxels, the summed probability of first_class and second_class will be used
    to create the segmentation map. If the summed probability of first_class and second_class is greater than
    external_class, we will label the voxel with first_class or second_class (whichever has the higher probability)
    instead of external_class.
    """
    first_class: str
    second_class: str
    external_class: str

    def validate(self, ground_truth_ids: List[str]) -> None:
        if self.first_class not in ground_truth_ids:
            raise ValueError(f"SummedProbabilityRule.validate: {self.first_class} not in ground truth IDs")
        if self.second_class not in ground_truth_ids:
            raise ValueError(f"SummedProbabilityRule.validate: {self.second_class} not in ground truth IDs")
        if self.external_class not in ground_truth_ids:
            raise ValueError(f"SummedProbabilityRule.validate: {self.external_class} not in ground truth IDs")


# The amount by which all Basic architectures shrink the input image.
basic_size_shrinkage = 28


def get_center_size(arch: str, crop_size: TupleInt3) -> TupleInt3:
    """
    Computes the size of the output tensor, if the model is fed with an input tensor of the given crop_size.
    This makes a lot of assumptions about the architectures that are hardcoded, this method should be used with care.

    :param arch: The model architecture that is used.
    :param crop_size: The size of the model's input tensor.
    :return: The size of the model's output tensor.
    """
    if arch == ModelArchitectureConfig.UNet3D or arch == ModelArchitectureConfig.UNet2D:
        return crop_size

    if arch in [ModelArchitectureConfig.Basic]:
        diff = basic_size_shrinkage
        return crop_size[0] - diff, crop_size[1] - diff, crop_size[2] - diff
    else:
        raise Exception("Unknown model architecture: {}".format(arch))


def equally_weighted_classes(foreground_classes: List[str], background_weight: Optional[float] = None) -> List:
    """
    Computes a list of weights for the background class and all foreground classes. If no background_weight
    is given, all foreground classes and the background class (class index 0) are given equal weight.
    If a background_weight is given explicitly, that weight is assigned to class index 0, and the rest of the weight
    is equally distributed across all foreground classes. All weights will sum to 1.0

    :param foreground_classes: The list of foreground classes that the model uses.
    :param background_weight: The weight that should be given to the background class (index 0). This can be None.
    :return: A list of length len(foreground_classes) + 1, with weights for all classes including the background class.
            The weights will sum to 1.0
    """
    num_foreground_classes = len(foreground_classes)
    if num_foreground_classes == 0:
        raise ValueError("No foreground class present.")
    if background_weight is None:
        num_classes_with_background = num_foreground_classes + 1
        return [1.0 / num_classes_with_background] * num_classes_with_background
    if background_weight < 0.0 or background_weight >= 1.0:
        raise ValueError(f"background_weight must be in the interval [0, 1), but got: {background_weight}")
    foreground_weight = (1.0 - background_weight) / num_foreground_classes
    return [background_weight] + [foreground_weight] * num_foreground_classes


class SegmentationModelBase(ModelConfigBase):
    """
    A class that holds all settings that are specific to segmentation models.
    """

    #: The segmentation model architecture to use.
    #: Valid options are defined at :class:`ModelArchitectureConfig`: 'Basic (DeepMedic)', 'UNet3D', 'UNet2D'
    architecture: str = param.String("Basic", doc="The model architecture (for example, UNet). Valid options are"
                                                  "UNet3D, UNet2D, Basic (DeepMedic)")

    #: The loss type to use during training.
    #: Valid options are defined at :class:`SegmentationLoss`: "SoftDice", "CrossEntropy", "Focal", "Mixture"
    loss_type: SegmentationLoss = param.ClassSelector(default=SegmentationLoss.SoftDice, class_=SegmentationLoss,
                                                      instantiate=False, doc="The loss_type to use")

    #: List of pairs of weights, loss types and class-weight-power values for use when loss_type is
    #: :attr:`SegmentationLoss.MixtureLoss`".
    mixture_loss_components: Optional[List[MixtureLossComponent]] = param.List(
        None, class_=MixtureLossComponent, instantiate=False,
        doc="List of pairs of weights, loss types and class-weight-power values for use when loss_type is MixtureLoss")

    #: For weighted loss, power to which to raise the weights per class. If this is None, loss is not weighted.
    loss_class_weight_power: Optional[float] = param.Number(None, allow_None=True,
                                                            doc="Power to which to raise class weights for loss "
                                                                "function; default value will depend on loss_type")

    #: Gamma value for focal loss: weight for each pixel is posterior likelihood to the power -focal_loss_gamma.
    focal_loss_gamma: float = param.Number(1.0, doc="Gamma value for focal loss: weight for each pixel is "
                                                    "posterior likelihood to the power -focal_loss_gamma.")

    #: The spacing X, Y, Z expected for all images in the dataset
    dataset_expected_spacing_xyz: Optional[TupleFloat3] = param.NumericTuple(
        None, length=3, allow_None=True,
        doc="The spacing X, Y, Z expected for all images in the dataset")

    #: The number of feature channels at different stages of the model.
    feature_channels: List[int] = param.List(None, class_=int, bounds=(1, None), instantiate=False,
                                             doc="The number of feature channels at different stages of the model.")

    #: The size of the convolution kernels.
    kernel_size: int = param.Integer(3, bounds=(1, None), doc="The size of the convolution kernels.")

    #: The size of the random crops that will be drawn from the input images during training. This is also the
    #: input size of the model.
    crop_size: TupleInt3 = IntTuple((1, 1, 1), length=3, doc="The size of the random crops that will be "
                                                             "drawn from the input images. This is also the "
                                                             "input size of the model.")

    #: The names of the image input channels that the model consumes. These channels must be present in the
    #: dataset.csv file.
    image_channels: List[str] = param.List(None, class_=str, bounds=(1, None), instantiate=False,
                                           doc="The names of the image input channels that the model consumes. "
                                               "These channels must be present in the dataset.csv file")

    #: The names of the ground truth channels that the model consumes. These channels must be present in the
    #: dataset.csv file
    ground_truth_ids: List[str] = param.List(None, class_=str, bounds=(1, None), instantiate=False,
                                             doc="The names of the ground truth channels that the model consumes. "
                                                 "These channels must be present in the dataset.csv file")

    #: The name of the channel that contains the `inside/outside body` information (to mask out the background).
    #: This channel must be present in the dataset
    mask_id: Optional[str] = param.String(None, allow_None=True, doc="The name of the channel that contains the "
                                                                     "`inside/outside body` information."
                                                                     "This channel must be present in the dataset")

    #: The type of image normalization that should be applied. Must be None, or of type
    # :attr:`PhotometricNormalizationMethod`: Unchanged, SimpleNorm, MriWindow , CtWindow, TrimmedNorm
    norm_method: PhotometricNormalizationMethod = \
        param.ClassSelector(default=PhotometricNormalizationMethod.CtWindow,
                            class_=PhotometricNormalizationMethod,
                            instantiate=False,
                            doc="The type of image normalization that should be applied. Must be one of None, "
                                "Unchanged, SimpleNorm, MriWindow , CtWindow, TrimmedNorm")

    #: The Window setting for the :attr:`PhotometricNormalizationMethod.CtWindow` normalization.
    window: int = param.Integer(600, bounds=(0, None), doc="The Window setting for the 'CtWindow' normalization.")

    #: The level setting for the :attr:`PhotometricNormalizationMethod.CtWindow` normalization.
    level: int = param.Integer(50, doc="The level setting for the 'CtWindow' normalization.")

    #: The value range that image normalization should produce. This is the input range to the network.
    output_range: TupleFloat2 = param.NumericTuple((-1.0, 1.0), length=2,
                                                   doc="The value range that image normalization should produce. "
                                                       "This is the input range to the network.")

    #: If true, create additional plots during image normalization.
    debug_mode: bool = param.Boolean(False, doc="If true, create additional plots during image normalization.")

    #: Tail parameter allows window range to be extended to right, used in
    #: :attr:`PhotometricNormalizationMethod.MriWindow`. The value must be a list with one entry per input channel
    #: if the model has multiple input channels
    tail: List[float] = param.List(None, class_=float,
                                   doc="Tail parameter allows window range to be extended to right, Used in MriWindow."
                                       " The value must be a list with one entry per input channel "
                                       "if the model has multiple input channels.")

    #: Sharpen parameter specifies number of standard deviations from mean to be included in window range.
    #: Used in :attr:`PhotometricNormalizationMethod.MriWindow`
    sharpen: float = param.Number(0.9, doc="Sharpen parameter specifies number of standard deviations "
                                           "from mean to be included in window range. Used in MriWindow")

    #: Percentile at which to trim input distribution prior to normalization. Used in
    #: :attr:`PhotometricNormalizationMethod.TrimmedNorm`
    trim_percentiles: TupleFloat2 = param.NumericTuple((1.0, 99.0), length=2,
                                                       doc="Percentile at which to trim input distribution prior "
                                                           "to normalization. Used in TrimmedNorm")

    #: Padding mode to use for training and inference. See :attr:`PaddingMode` for valid options.
    padding_mode: PaddingMode = param.ClassSelector(default=PaddingMode.Edge, class_=PaddingMode,
                                                    instantiate=False,
                                                    doc="Padding mode to use for training and inference")

    #: The batch size to use for inference forward pass.
    inference_batch_size: int = param.Integer(8, bounds=(1, None),
                                              doc="The batch size to use for inference forward pass")

    #: The crop size to use for model testing. If nothing is specified, crop_size parameter is used instead,
    #: i.e. training and testing crop size will be the same.
    test_crop_size: Optional[TupleInt3] = IntTuple(None, length=3, allow_None=True,
                                                   doc="The crop size to use for model testing. "
                                                       "If nothing is specified, "
                                                       "crop_size parameter is used instead, "
                                                       "i.e. training and testing crop size "
                                                       "will be the same.")

    #: The per-class probabilities for picking a center point of a crop.
    class_weights: Optional[List[float]] = param.List(None, class_=float, bounds=(1, None), allow_None=True,
                                                      instantiate=False,
                                                      doc="The per-class probabilities for picking a center point of "
                                                          "a crop.")

    #: Layer name hierarchy (parent, child recursive) as by model definition. If None, no activation maps will be saved
    activation_map_layers: Optional[List[str]] = param.List(None, class_=str, allow_None=True, bounds=(1, None),
                                                            instantiate=False,
                                                            doc="Layer name hierarchy (parent, child "
                                                                "recursive) as by model definition. If None, "
                                                                "no activation maps will be saved")

    #: The aggregation method to use when testing ensemble models. See :attr: `EnsembleAggregationType` for options.
    ensemble_aggregation_type: EnsembleAggregationType = param.ClassSelector(default=EnsembleAggregationType.Average,
                                                                             class_=EnsembleAggregationType,
                                                                             instantiate=False,
                                                                             doc="The aggregation method to use when"
                                                                                 "testing ensemble models.")

    #: The size of the smoothing kernel in mm to be used for smoothing posteriors before computing the final
    #: segmentations. No smoothing is performed if set to None.
    posterior_smoothing_mm: Optional[TupleInt3] = param.NumericTuple(None, length=3, allow_None=True,
                                                                     doc="The size of the smoothing kernel in mm to be "
                                                                         "used for smoothing posteriors before "
                                                                         "computing the final segmentations. No "
                                                                         "smoothing is performed if set to None")

    #: If True save image and segmentations for one image in a batch for each training epoch
    store_dataset_sample: bool = param.Boolean(False, doc="If True save image and segmentations for one image"
                                                          "in a batch for each training epoch")

    #: List of (name, container) pairs, where name is a descriptive name and container is a Azure ML storage account
    #: container name to be used for statistical comparisons
    comparison_blob_storage_paths: List[Tuple[str, str]] = param.List(
        None, class_=tuple,
        allow_None=True,
        doc="List of (name, container) pairs, where name is a descriptive name and container is a "
            "Azure ML storage account container name to be used for statistical comparisons")

    #: List of rules for structures that should be prevented from sharing the same slice.
    #: These are not applied if :attr:`disable_extra_postprocessing` is True.
    #: Parameter should be a list of :attr:`SliceExclusionRule` objects.
    slice_exclusion_rules: List[SliceExclusionRule] = param.List(
        default=[], class_=SliceExclusionRule, allow_None=False,
        doc="List of rules for structures that should be prevented from sharing the same slice; "
            "not applied if disable_extra_postprocessing is True.")

    #: List of rules for class pairs whose summed probability is used to create the segmentation map from predicted
    #: posterior probabilities.
    #: These are not applied if :attr:`disable_extra_postprocessing` is True.
    #: Parameter should be a list of :attr:`SummedProbabilityRule` objects.
    summed_probability_rules: List[SummedProbabilityRule] = param.List(
        default=[], class_=SummedProbabilityRule, allow_None=False,
        doc="List of rules for class pairs whose summed probability is used to create the segmentation map from "
            "predicted posterior probabilities; not applied if disable_extra_postprocessing is True.")

    #: Whether to ignore :attr:`slice_exclusion_rules` and :attr:`summed_probability_rules` even if defined
    disable_extra_postprocessing: bool = param.Boolean(
        False, doc="Whether to ignore slice_exclusion_rules and summed_probability_rules even if defined")

    #: User friendly display names to be used for each of the predicted GT classes. Default is ground_truth_ids if
    #: None provided
    ground_truth_ids_display_names: List[str] = param.List(None, class_=str, bounds=(1, None), instantiate=False,
                                                           allow_None=True,
                                                           doc="User friendly display names to be used for each of "
                                                               "the predicted GT classes. Default is ground_truth_ids "
                                                               "if None provided")

    #: Colours in (R, G, B) for the structures, same order as in ground_truth_ids_display_names
    colours: List[TupleInt3] = param.List(None, class_=tuple, bounds=(1, None), instantiate=False,
                                          allow_None=True,
                                          doc="Colours in (R, G, B) for the structures, same order as in "
                                              "ground_truth_ids_display_names")

    #: List of bool specifiying if structures need filling holes. If True, the output of the model for that class
    #: will include postprocessing to fill holes, in the same order as in ground_truth_ids_display_names
    fill_holes: List[bool] = param.List(None, class_=bool, bounds=(1, None), instantiate=False,
                                        allow_None=True,
                                        doc="List of bool specifiying if structures need filling holes. If True "
                                            "output of the model for that class includes postprocessing to fill holes, "
                                            "in the same order as in ground_truth_ids_display_names")

    _inference_stride_size: Optional[TupleInt3] = IntTuple(None, length=3, allow_None=True,
                                                           doc="The stride size in the inference pipeline. "
                                                               "At most, this should be the output_size to "
                                                               "avoid gaps in output posterior image. If it "
                                                               "is not specified, its value is set to "
                                                               "output size.")
    _center_size: Optional[TupleInt3] = IntTuple(None, length=3, allow_None=True)
    _train_output_size: Optional[TupleInt3] = IntTuple(None, length=3, allow_None=True)
    _test_output_size: Optional[TupleInt3] = IntTuple(None, length=3, allow_None=True)

    #: Dictionary of types to enforce for certain DataFrame columns, where key is column name and value is desired type.
    col_type_converters: Optional[Dict[str, Any]] = param.Dict(None,
                                                               doc="Dictionary of types to enforce for certain "
                                                                   "DataFrame columns, where key is column name "
                                                                   "and value is desired type.",
                                                               allow_None=True, instantiate=False)

    _largest_connected_component_foreground_classes: LARGEST_CC_TYPE = \
        param.List(None, class_=None, bounds=(1, None), instantiate=False, allow_None=True,
                   doc="The names of the ground truth channels for which to select the largest connected component in "
                       "the model predictions as an inference post-processing step. Alternatively, a member of the "
                       "list can be a tuple (name, threshold), where name is a channel name and threshold is a value "
                       "between 0 and 0.5 such that disconnected components will be kept if their volume (relative "
                       "to the whole structure) exceeds that value.")

    #: If true, various overview plots with results are generated during model evaluation. Set to False if you see
    #: non-deterministic pull request build failures.
    is_plotting_enabled: bool = param.Boolean(True, doc="If true, various overview plots with results are generated "
                                                        "during model evaluation. Set to False if you see "
                                                        "non-deterministic pull request build failures.")
    show_patch_sampling: int = param.Integer(5, bounds=(0, None),
                                             doc="Number of patients from the training set for which the effect of"
                                                 "patch sampling will be shown. Nifti images and thumbnails for each"
                                                 "of the first N subjects in the training set will be "
                                                 "written to the outputs folder.")

    def __init__(self, center_size: Optional[TupleInt3] = None,
                 inference_stride_size: Optional[TupleInt3] = None,
                 min_l_rate: float = 0,
                 largest_connected_component_foreground_classes: LARGEST_CC_TYPE = None,
                 **params: Any):
        super().__init__(**params)
        self.test_crop_size = self.test_crop_size if self.test_crop_size is not None else self.crop_size
        self.inference_stride_size = inference_stride_size
        self.min_l_rate = min_l_rate
        self.largest_connected_component_foreground_classes = largest_connected_component_foreground_classes
        self._center_size = center_size
        self._model_category = ModelCategory.Segmentation

    def validate(self) -> None:
        """
        Validates the parameters stored in the present object.
        """
        super().validate()
        check_is_any_of("Architecture", self.architecture, vars(ModelArchitectureConfig).keys())

        def len_or_zero(lst: Optional[List[Any]]) -> int:
            return 0 if lst is None else len(lst)

        if self.kernel_size % 2 == 0:
            raise ValueError("The kernel size must be an odd number (kernel_size: {})".format(self.kernel_size))

        if self.architecture != ModelArchitectureConfig.UNet3D:
            if any_pairwise_larger(self.center_size, self.crop_size):
                raise ValueError("Each center_size should be less than or equal to the crop_size "
                                 "(center_size: {}, crop_size: {}".format(self.center_size, self.crop_size))
        else:
            if self.crop_size != self.center_size:
                raise ValueError("For UNet3D, the center size of each dimension should be equal to the crop size "
                                 "(center_size: {}, crop_size: {}".format(self.center_size, self.crop_size))

        self.validate_inference_stride_size(self.inference_stride_size, self.get_output_size())

        # check to make sure there is no overlap between image and ground-truth channels
        image_gt_intersect = np.intersect1d(self.image_channels, self.ground_truth_ids)
        if len(image_gt_intersect) != 0:
            raise ValueError("Channels: {} were found in both image_channels, and ground_truth_ids"
                             .format(image_gt_intersect))

        valid_norm_methods = [method.value for method in PhotometricNormalizationMethod]
        check_is_any_of("norm_method", self.norm_method.value, valid_norm_methods)

        if len(self.trim_percentiles) < 2 or self.trim_percentiles[0] >= self.trim_percentiles[1]:
            raise ValueError("Thresholds should contain lower and upper percentile thresholds, but got: {}"
                             .format(self.trim_percentiles))

        if len_or_zero(self.class_weights) != (len_or_zero(self.ground_truth_ids) + 1):
            raise ValueError("class_weights needs to be equal to number of ground_truth_ids + 1")
        if self.class_weights is None:
            raise ValueError("class_weights must be set.")
        SegmentationModelBase.validate_class_weights(self.class_weights)
        if self.ground_truth_ids is None:
            raise ValueError("ground_truth_ids is None")
        if len(self.ground_truth_ids_display_names) != len(self.ground_truth_ids):
            raise ValueError("len(ground_truth_ids_display_names)!=len(ground_truth_ids)")
        if len(self.ground_truth_ids_display_names) != len(self.colours):
            raise ValueError("len(ground_truth_ids_display_names)!=len(colours)")
        if len(self.ground_truth_ids_display_names) != len(self.fill_holes):
            raise ValueError("len(ground_truth_ids_display_names)!=len(fill_holes)")
        if self.mean_teacher_alpha is not None:
            raise ValueError("Mean teacher model is currently only supported for ScalarModels."
                             "Please reset mean_teacher_alpha to None.")

    @staticmethod
    def validate_class_weights(class_weights: List[float]) -> None:
        """
        Checks that the given list of class weights is valid: The weights must be positive and add up to 1.0.
        Raises a ValueError if that is not the case.
        """
        if not isclose(sum(class_weights), 1.0):
            raise ValueError(f'class_weights needs to add to 1 but it was: {sum(class_weights)}')
        if np.any(np.array(class_weights) < 0):
            raise ValueError("class_weights must have non-negative values only, found: {}".format(class_weights))

    @staticmethod
    def validate_inference_stride_size(inference_stride_size: Optional[TupleInt3],
                                       output_size: Optional[TupleInt3]) -> None:
        """
        Checks that patch stride size is positive and smaller than output patch size to ensure that posterior
        predictions are obtained for all pixels
        """
        if inference_stride_size is not None:
            if any_smaller_or_equal_than(inference_stride_size, 0):
                raise ValueError("inference_stride_size must be > 0 in all dimensions, found: {}"
                                 .format(inference_stride_size))

            if output_size is not None:
                if any_pairwise_larger(inference_stride_size, output_size):
                    raise ValueError("inference_stride_size must be <= output_size in all dimensions"
                                     "Found: output_size={}, inference_stride_size={}"
                                     .format(output_size, inference_stride_size))

    @property
    def number_of_image_channels(self) -> int:
        """
        Gets the number of image input channels that the model has (usually 1 CT channel, or multiple MR).
        """
        return 0 if self.image_channels is None else len(self.image_channels)

    @property
    def number_of_classes(self) -> int:
        """
        Returns the number of ground truth ids, including the background class.
        """
        return 1 if self.ground_truth_ids is None else len(self.ground_truth_ids) + 1

    @property
    def center_size(self) -> TupleInt3:
        """
        Gets the size of the center crop that the model predicts.
        """
        if self._center_size is None:
            return get_center_size(arch=self.architecture, crop_size=self.crop_size)
        Warning("'center_size' argument will soon be deprecated. Output shapes are inferred from models on the fly.")
        return self._center_size

    @property
    def inference_stride_size(self) -> Optional[TupleInt3]:
        """
        Gets the stride size that should be used when stitching patches at inference time.
        """
        if self._inference_stride_size is None:
            return self.get_output_size(ModelExecutionMode.TEST)
        return self._inference_stride_size

    @inference_stride_size.setter
    def inference_stride_size(self, val: Optional[TupleInt3]) -> None:
        """
        Sets the inference stride size with given value. This setter is used if output shape needs to be
        determined dynamically at run time
        """
        self._inference_stride_size = val
        self.validate_inference_stride_size(inference_stride_size=val,
                                            output_size=self.get_output_size(ModelExecutionMode.TEST))

    @property
    def example_images_folder(self) -> Path:
        """
        Gets the full path in which the example images should be stored during training.
        """
        return self.outputs_folder / EXAMPLE_IMAGES_FOLDER

    @property
    def largest_connected_component_foreground_classes(self) -> LARGEST_CC_TYPE:
        """
        Gets the list of classes for which the largest connected components should be computed when predicting.
        """
        return self._largest_connected_component_foreground_classes

    @largest_connected_component_foreground_classes.setter
    def largest_connected_component_foreground_classes(self, value: LARGEST_CC_TYPE) -> None:
        """
        Sets the list of classes for which the largest connected components should be computed when predicting.
        """
        pairs: Optional[List[Tuple[str, Optional[float]]]] = None
        if value is not None:
            # Set all members to be tuples rather than just class names.
            pairs = [val if isinstance(val, tuple) else (val, None) for val in value]
            class_names = set(pair[0] for pair in pairs)
            unknown_labels = class_names - set(self.ground_truth_ids)
            if unknown_labels:
                raise ValueError(
                    f"Found unknown labels {unknown_labels} in largest_connected_component_foreground_classes: "
                    f"labels must exist in [{self.ground_truth_ids}]")
            bad_thresholds = [pair[1] for pair in pairs if (pair[1] is not None)
                              and (pair[1] <= 0.0 or pair[1] > 0.5)]  # type: ignore
            if bad_thresholds:
                raise ValueError(
                    f"Found bad threshold(s) {bad_thresholds} in largest_connected_component_foreground_classes: "
                    "thresholds must be positive and at most 0.5.")

        self._largest_connected_component_foreground_classes = pairs

    def read_dataset_into_dataframe_and_pre_process(self) -> None:
        """
        Loads a dataset from the dataset.csv file, and stores it in the present object.
        """
        assert self.local_dataset is not None  # for mypy
        self.dataset_data_frame = pd.read_csv(self.local_dataset / DATASET_CSV_FILE_NAME,
                                              dtype=str,
                                              converters=self.col_type_converters,
                                              low_memory=False)
        self.pre_process_dataset_dataframe()

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        """
        Turns the given AzureML estimator (settings for running a job in AzureML) into a configuration object
        for doing hyperparameter searches.

        :param estimator: The settings for running a single AzureML job.
        :return: A HyperDriveConfig object for running multiple AzureML jobs.
        """
        return super().get_parameter_search_hyperdrive_config(estimator)

    def get_model_train_test_dataset_splits(self, dataset_df: DataFrame) -> DatasetSplits:
        """
        Computes the training, validation and test splits for the model, from a dataframe that contains
        the full dataset.

        :param dataset_df: A dataframe that contains the full dataset that the model is using.
        :return: An instance of DatasetSplits with dataframes for training, validation and testing.
        """
        return super().get_model_train_test_dataset_splits(dataset_df)

    def get_output_size(self, execution_mode: ModelExecutionMode = ModelExecutionMode.TRAIN) -> Optional[TupleInt3]:
        """
        Returns shape of model's output tensor for training, validation and testing inference modes
        """
        if (execution_mode == ModelExecutionMode.TRAIN) or (execution_mode == ModelExecutionMode.VAL):
            return self._train_output_size
        elif execution_mode == ModelExecutionMode.TEST:
            return self._test_output_size
        raise ValueError("Unknown execution mode '{}' for function 'get_output_size'".format(execution_mode))

    def adjust_after_mixed_precision_and_parallel(self, model: Any) -> None:
        """
        Updates the model config parameters (e.g. output patch size). If testing patch stride size is unset then
        its value is set by the output patch size
        """
        self._train_output_size = model.get_output_shape(input_shape=self.crop_size)
        self._test_output_size = model.get_output_shape(input_shape=self.test_crop_size)
        if self.inference_stride_size is None:
            self.inference_stride_size = self._test_output_size
        else:
            if any_pairwise_larger(self.inference_stride_size, self._test_output_size):
                raise ValueError("The inference stride size must be smaller than the model's output size in each"
                                 "dimension. Inference stride was set to {}, the model outputs {} in test mode."
                                 .format(self.inference_stride_size, self._test_output_size))

    def class_and_index_with_background(self) -> Dict[str, int]:
        """
        Returns a dict of class names to indices, including the background class.
        The class index assumes that background is class 0, foreground starts at 1.
        For example, if the ground_truth_ids are ["foo", "bar"], the result
        is {"background": 0, "foo": 1, "bar": 2}

        :return: A dict, one entry for each entry in ground_truth_ids + 1 for the background class.
        """
        classes = {BACKGROUND_CLASS_NAME: 0}
        classes.update({x: i + 1 for i, x in enumerate(self.ground_truth_ids)})
        return classes

    def create_and_set_torch_datasets(self, for_training: bool = True, for_inference: bool = True) -> None:
        """
        Creates torch datasets for all model execution modes, and stores them in the object.
        """
        from InnerEye.ML.dataset.cropping_dataset import CroppingDataset
        from InnerEye.ML.dataset.full_image_dataset import FullImageDataset

        dataset_splits = self.get_dataset_splits()
        crop_transforms = self.get_cropped_image_sample_transforms()
        full_image_transforms = self.get_full_image_sample_transforms()
        if for_training:
            self._datasets_for_training = {
                ModelExecutionMode.TRAIN: CroppingDataset(
                    self,
                    dataset_splits.train,
                    cropped_sample_transforms=crop_transforms.train,  # type: ignore
                    full_image_sample_transforms=full_image_transforms.train),  # type: ignore
                ModelExecutionMode.VAL: CroppingDataset(
                    self, dataset_splits.val,
                    cropped_sample_transforms=crop_transforms.val,  # type: ignore
                    full_image_sample_transforms=full_image_transforms.val),  # type: ignore
            }
        if for_inference:
            self._datasets_for_inference = {
                mode: FullImageDataset(
                    self,
                    dataset_splits[mode],
                    full_image_sample_transforms=full_image_transforms.test)  # type: ignore
                for mode in ModelExecutionMode if len(dataset_splits[mode]) > 0
            }

    def create_model(self) -> Any:
        """
        Creates a PyTorch model from the settings stored in the present object.

        :return: The network model as a torch.nn.Module object
        """
        # Use a local import here to avoid reliance on pytorch too early.
        # Return type should be BaseModel, but that would also introduce reliance on pytorch.
        from InnerEye.ML.utils.model_util import build_net
        return build_net(self)

    def get_full_image_sample_transforms(self) -> ModelTransformsPerExecutionMode:
        """
        Get transforms to perform on full image samples for each model execution mode.
        By default only PhotometricNormalization is performed.
        """
        from InnerEye.ML.utils.transforms import Compose3D
        from InnerEye.ML.photometric_normalization import PhotometricNormalization

        photometric_transformation = Compose3D(transforms=[PhotometricNormalization(self, use_gpu=False)])
        return ModelTransformsPerExecutionMode(train=photometric_transformation,
                                               val=photometric_transformation,
                                               test=photometric_transformation)

    def get_cropped_image_sample_transforms(self) -> ModelTransformsPerExecutionMode:
        """
        Get transforms to perform on cropped samples for each model execution mode.
        By default no transformation is performed.
        """
        return ModelTransformsPerExecutionMode()

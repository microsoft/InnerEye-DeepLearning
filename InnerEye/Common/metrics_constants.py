#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum, unique
# String prefixes when writing training or validation set metrics to a logger
from typing import Union

TRAIN_PREFIX = "train/"
VALIDATION_PREFIX = "val/"

SEQUENCE_POSITION_HUE_NAME_PREFIX = "Seq_pos"
# Suffix for logging a Dice score averaged across all structures
AVERAGE_DICE_SUFFIX = "AverageAcrossStructures"


class MetricsFileColumns(Enum):
    """
    Contains the names of the columns in the CSV file that is written by model testing.
    """
    Patient = "Patient"
    Structure = "Structure"
    Dice = "Dice"
    DiceNumeric = "DiceNumeric"
    HausdorffDistanceMM = "HausdorffDistance_mm"
    MeanDistanceMM = "MeanDistance_mm"


@unique
class LoggingColumns(Enum):
    """
    This enum contains string constants that act as column names in logging, and in all files on disk.
    """
    DataSplit = "data_split"
    Patient = "subject"
    Hue = "prediction_target"
    Structure = "structure"
    Dice = "dice"
    HausdorffDistanceMM = "HausdorffDistanceMM"
    Epoch = "epoch"
    Institution = "institutionId"
    Series = "seriesId"
    Tags = "tags"
    AccuracyAtThreshold05 = "accuracy_at_threshold_05"
    Loss = "loss"
    CrossEntropy = "cross_entropy"
    AreaUnderRocCurve = "area_under_roc_curve"
    AreaUnderPRCurve = "area_under_pr_curve"
    CrossValidationSplitIndex = "cross_validation_split_index"
    ModelOutput = "model_output"
    Label = "label"
    SubjectCount = "subject_count"
    ModelExecutionMode = "model_execution_mode"
    MeanAbsoluteError = "mean_absolute_error"
    MeanSquaredError = "mean_squared_error"
    LearningRate = "learning_rate"
    ExplainedVariance = "explained_variance"
    NumTrainableParameters = "num_trainable_parameters"
    AccuracyAtOptimalThreshold = "accuracy_at_optimal_threshold"
    OptimalThreshold = "optimal_threshold"
    FalsePositiveRateAtOptimalThreshold = "false_positive_rate_at_optimal_threshold"
    FalseNegativeRateAtOptimalThreshold = "false_negative_rate_at_optimal_threshold"
    SequenceLength = "sequence_length"


@unique
class MetricType(Enum):
    """
    Contains the different metrics that are computed.
    """
    # Any result of loss computation, depending on what's configured in the model.
    LOSS = "Loss"

    # Classification metrics
    CROSS_ENTROPY = "CrossEntropy"
    # Classification accuracy assuming that posterior > 0.5 means predicted class 1
    ACCURACY_AT_THRESHOLD_05 = "AccuracyAtThreshold05"
    ACCURACY_AT_OPTIMAL_THRESHOLD = "AccuracyAtOptimalThreshold"
    # Metrics for segmentation
    DICE = "Dice"
    HAUSDORFF_mm = "HausdorffDistance_millimeters"
    MEAN_SURFACE_DIST_mm = "MeanSurfaceDistance_millimeters"
    VOXEL_COUNT = "VoxelCount"
    PROPORTION_FOREGROUND_VOXELS = "ProportionForegroundVoxels"

    PATCH_CENTER = "PatchCenter"

    AREA_UNDER_ROC_CURVE = "AreaUnderRocCurve"
    AREA_UNDER_PR_CURVE = "AreaUnderPRCurve"
    OPTIMAL_THRESHOLD = "OptimalThreshold"
    FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalsePositiveRateAtOptimalThreshold"
    FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD = "FalseNegativeRateAtOptimalThreshold"

    # Regression metrics
    MEAN_ABSOLUTE_ERROR = "MeanAbsoluteError"
    MEAN_SQUARED_ERROR = "MeanSquaredError"
    EXPLAINED_VAR = "ExplainedVariance"

    # Common metrics
    SUBJECT_COUNT = "SubjectCount"
    LEARNING_RATE = "LearningRate"


MetricTypeOrStr = Union[str, MetricType]

# Mapping from the internal logging column names to the ones used in the outside-facing pieces of code:
# Output data files, logging systems.
INTERNAL_TO_LOGGING_COLUMN_NAMES = {
    MetricType.LOSS.value: LoggingColumns.Loss,
    MetricType.ACCURACY_AT_THRESHOLD_05.value: LoggingColumns.AccuracyAtThreshold05,
    MetricType.CROSS_ENTROPY.value: LoggingColumns.CrossEntropy,
    MetricType.AREA_UNDER_ROC_CURVE.value: LoggingColumns.AreaUnderRocCurve,
    MetricType.AREA_UNDER_PR_CURVE.value: LoggingColumns.AreaUnderPRCurve,
    MetricType.SUBJECT_COUNT.value: LoggingColumns.SubjectCount,
    MetricType.MEAN_SQUARED_ERROR.value: LoggingColumns.MeanSquaredError,
    MetricType.MEAN_ABSOLUTE_ERROR.value: LoggingColumns.MeanAbsoluteError,
    MetricType.EXPLAINED_VAR.value: LoggingColumns.ExplainedVariance,
    MetricType.LEARNING_RATE.value: LoggingColumns.LearningRate,
    MetricType.ACCURACY_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.AccuracyAtOptimalThreshold,
    MetricType.OPTIMAL_THRESHOLD.value: LoggingColumns.OptimalThreshold,
    MetricType.FALSE_POSITIVE_RATE_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.FalsePositiveRateAtOptimalThreshold,
    MetricType.FALSE_NEGATIVE_RATE_AT_OPTIMAL_THRESHOLD.value: LoggingColumns.FalseNegativeRateAtOptimalThreshold
}


class TrackedMetrics(Enum):
    """
    Known metrics that are tracked as part of Hyperdrive runs.
    """
    Val_Loss = VALIDATION_PREFIX + MetricType.LOSS.value

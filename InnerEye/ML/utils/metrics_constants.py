#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum, unique


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
    SecondsPerEpoch = "seconds_per_epoch"
    SecondsPerBatch = "seconds_per_batch"
    AreaUnderRocCurve = "area_under_roc_curve"
    AreaUnderPRCurve = "area_under_pr_curve"
    CrossValidationSplitIndex = "cross_validation_split_index"
    ModelOutput = "model_output"
    Label = "label"
    SubjectCount = "subject_count"
    ModelExecutionMode = "model_execution_mode"
    PredictedValue = "predicted_value"
    MeanAbsoluteError = "mean_absolute_error"
    MeanSquaredError = "mean_squared_error"
    LearningRate = "learning_rate"
    R2Score = "r2_score"
    NumTrainableParameters = "num_trainable_parameters"
    AccuracyAtOptimalThreshold = "accuracy_at_optimal_threshold"
    OptimalThreshold = "optimal_threshold"
    FalsePositiveRateAtOptimalThreshold = "false_positive_rate_at_optimal_threshold"
    FalseNegativeRateAtOptimalThreshold = "false_negative_rate_at_optimal_threshold"
    SequenceLength = "sequence_length"

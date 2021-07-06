#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from enum import Enum, unique
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import param
import torch
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import HyperDriveConfig

from torch.nn import ModuleDict, ModuleList

from InnerEye.Common.common_util import print_exception
from InnerEye.Common.generic_parsing import ListOrDictParam
from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.Common.type_annotations import TupleInt3

from InnerEye.ML.common import ModelExecutionMode, OneHotEncoderBase
from InnerEye.ML.deep_learning_config import ModelCategory
from InnerEye.ML.lightning_metrics import Accuracy05, AccuracyAtOptimalThreshold, AreaUnderPrecisionRecallCurve, \
    AreaUnderRocCurve, BinaryCrossEntropyWithLogits, ExplainedVariance, FalseNegativeRateOptimalThreshold, \
    FalsePositiveRateOptimalThreshold, MeanAbsoluteError, MeanSquaredError, OptimalThreshold, ScalarMetricsBase
from InnerEye.ML.metrics_dict import DEFAULT_KEY, DataframeLogger
from InnerEye.ML.model_config_base import ModelConfigBase, ModelTransformsPerExecutionMode
from InnerEye.ML.utils.csv_util import CSV_CHANNEL_HEADER, CSV_SUBJECT_HEADER
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels


class AggregationType(Enum):
    """
    The type of global pooling aggregation to use between the encoder and the classifier.
    """
    ZAdaptive3dAvg = "Adaptive3dAverage"
    Average = "Average"
    GatedPooling = "Gated"
    MixPooling = "Mix"
    MaxPooling = "Max"


class EnsembleAggregationType(Enum):
    Average = "Average"


@unique
class ScalarLoss(Enum):
    BinaryCrossEntropyWithLogits = "BinaryCrossEntropyWithLogits"
    WeightedCrossEntropyWithLogits = "WeightedCrossEntropyWithLogits"
    MeanSquaredError = "MeanSquaredError"
    CustomClassification = "CustomClassification"
    CustomRegression = "CustomRegression"

    def is_classification_loss(self) -> bool:
        return self in {self.BinaryCrossEntropyWithLogits, self.WeightedCrossEntropyWithLogits,
                        self.CustomClassification}

    def is_regression_loss(self) -> bool:
        return self in {self.MeanSquaredError, self.CustomRegression}


@unique
class LabelTransformation(Enum):
    """
    Defines all possible transformation to apply to the labels.
    """

    @staticmethod
    def identity(labels: List) -> Any:
        """
        No changes applied, checks that only one channel has been provided.
        """
        assert len(labels) == 1
        return labels[0]

    @staticmethod
    def difference(labels: List) -> Any:
        """
        Assumes two labels channels [chan_1, chan_2] as input, the returned
        labels is the difference chan_2 -  chan_1.
        """
        assert len(labels) == 2
        return labels[1] - labels[0]

    @staticmethod
    def get_scaling_transform(max_value: int = 100, min_value: int = 0, last_in_pipeline: bool = True) -> Callable:
        """
        Defines the function to scale labels.
        :param max_value:
        :param min_value:
        :param last_in_pipeline: if the transformation is the last
        in the pipeline it should expect a single label as an argument.
        Else if returns a list of scaled labels for further transforms.
        :return: The scaling function
        """

        def scale(labels: List) -> Union[List, Any]:
            result = []
            for label in labels:
                result.append((label - min_value) / max_value)
            if last_in_pipeline:
                # if there is no further transform it
                # should be only one label
                assert len(labels) == 1
                return result[0]
            else:
                return result

        return scale


class ScalarModelBase(ModelConfigBase):
    class_names: List[str] = param.List(class_=str,
                                        default=[DEFAULT_KEY],
                                        bounds=(1, None),
                                        doc="The label names for each label class in the dataset and model output "
                                            "in the case of binary and multi-label classification tasks."
                                            "The order of the names should match the order of label class indices "
                                            "in dataset.csv"
                                            "For multi-label classification, this field is required."
                                            "For binary classification, this field must be a list of size 1, and "
                                            "is by default ['Default'], but can optionally be set to a more "
                                            "descriptive "
                                            "name for the positive class.")
    target_names: List[str] = param.List(class_=str,
                                         default=None,
                                         bounds=(1, None),
                                         doc="The label names for each output target, used for logging metrics and "
                                             "reporting results. If provided, the length of this list must match the "
                                             "number of model outputs (and of transformed labels, if defined; see "
                                             "get_posthoc_label_transform()). By default, this inherits the value of "
                                             "class_names at initialisation.")
    aggregation_type: AggregationType = param.ClassSelector(default=AggregationType.Average, class_=AggregationType,
                                                            doc="The type of global pooling aggregation to use between"
                                                                " the encoder and the classifier.")
    loss_type: ScalarLoss = param.ClassSelector(default=ScalarLoss.BinaryCrossEntropyWithLogits, class_=ScalarLoss,
                                                instantiate=False, doc="The loss_type to use")
    image_channels: List[str] = param.List(class_=str,
                                           doc="Identifies the rows of the dataset file that contain image file paths.")
    image_file_column: Optional[str] = param.String(default=None, allow_None=True,
                                                    doc="The column that contains the path to image files.")
    expected_column_values: List[Tuple[str, str]] = \
        param.List(default=None,
                   doc="List of tuples with column name and expected value to filter rows in the dataset csv file",
                   allow_None=True)
    label_channels: Optional[List[str]] = \
        param.List(default=None, allow_None=True,
                   doc="Identifies the row of a dataset file that contains the label value.")
    label_value_column: str = param.String(doc="The column in the dataset file that contains the label value.")
    non_image_feature_channels: Union[List[str], Dict[str, List[str]]] = \
        ListOrDictParam(doc="Specifies the rows of a dataset file from which additional feature values should be read."
                            "The channels can be specified as a List of channels to be used for all non imaging"
                            "features or a as Dict mapping features to specific channels. The helper function"
                            "`get_non_image_features_dict` is available to construct this dictionnary.")
    numerical_columns: List[str] = \
        param.List(class_=str,
                   default=[],
                   doc="Specifies the columns of a dataset file from which additional numerical "
                       "feature values should be read.")
    categorical_columns: List[str] = \
        param.List(class_=str,
                   default=[],
                   doc="Specifies the columns of a dataset file from which additional "
                       "catagorical feature values should be read.")

    subject_column: str = \
        param.String(default=CSV_SUBJECT_HEADER, allow_None=False,
                     doc="The name of the column that contains the patient/subject identifier. Default: 'subject'")
    channel_column: str = \
        param.String(default=CSV_CHANNEL_HEADER, allow_None=False,
                     doc="The name of the column that contains image channel information, for identifying multiple "
                         "rows belonging to the same subject. Default: 'channel'")

    add_differences_for_features: List[str] = \
        param.List(class_=str,
                   doc="When using sequence datasets, this specifies the columns in the dataset for which additional"
                       "features should be added. For all columns given here, the feature differences between index i"
                       "and index 0 (start of the sequence) are added as additional features.")
    traverse_dirs_when_loading: bool = \
        param.Boolean(doc="If true, image file names in datasets do no need to contain "
                          "the full path. Before loading, all files will be enumerated "
                          "recursively. If false, the image file name must be fully "
                          "given in the dataset file (relative to root path)")
    load_segmentation: bool = \
        param.Boolean(default=False, doc="If True the segmentations from hdf5 files will be loaded. If False, only"
                                         "the images will be loaded.")
    center_crop_size: Optional[TupleInt3] = \
        param.NumericTuple(default=None, allow_None=True, length=3,
                           doc="If given, the loaded images and segmentations will be cropped to the given size."
                               "Size is given in pixels. The crop will be taken from the center of the image. "
                               "Crop size should be in the form (crop_z, crop_y, crop_x)."
                               "If your dataset has 2D images, center crop should have singleton first dimension,"
                               "i.e. (1, ) + (crop_y, crop_x)")

    image_size: Optional[TupleInt3] = \
        param.NumericTuple(default=None, allow_None=True, length=3,
                           doc="If given, images will be resized to these dimensions immediately after loading from"
                               "file."
                               "Image size should be in the form (size_z, size_y, size_x)."
                               "If your dataset has 2D images, image size should have singleton first dimension,"
                               "i.e. (1, ) + (size_y, size_x)")

    categorical_feature_encoder: Optional[OneHotEncoderBase] = param.ClassSelector(OneHotEncoderBase,
                                                                                   allow_None=True,
                                                                                   instantiate=False,
                                                                                   doc="The one hot encoding scheme "
                                                                                       "for categorical data if "
                                                                                       "required")
    num_dataset_reader_workers: int = param.Integer(default=0, bounds=(-1, None),
                                                    doc="Number of workers (processes) to use for dataset "
                                                        "reading. Default is 0 which means only the main thread "
                                                        "will be used. Set to -1 for maximum parallelism level.")

    ensemble_aggregation_type: EnsembleAggregationType = param.ClassSelector(default=EnsembleAggregationType.Average,
                                                                             class_=EnsembleAggregationType,
                                                                             instantiate=False,
                                                                             doc="The aggregation method to use when"
                                                                                 "testing ensemble models.")

    dataset_stats_hook: Optional[Callable[[Dict[ModelExecutionMode, Any]], None]] = \
        param.Callable(default=None,
                       allow_None=True,
                       doc="A hook that is called with a dictionary that maps from train/val/test to the actual "
                           "dataset, to do customized statistics.")

    def __init__(self, num_dataset_reader_workers: int = 0, **params: Any) -> None:
        super().__init__(**params)
        self._model_category = ModelCategory.Regression \
            if self.is_regression_model else ModelCategory.Classification
        if not self.is_offline_run:
            self.num_dataset_reader_workers = 0
            logging.info("dataset reader parallelization is supported only locally, setting "
                         "num_dataset_reader_workers to 0 as this is an AML run.")
        else:
            self.num_dataset_reader_workers = num_dataset_reader_workers
        if self.target_names is None:
            self.target_names = self.class_names

    def validate(self) -> None:
        if len(self.class_names) > 1 and not self.is_classification_model:
            raise ValueError("Multiple label classes supported only for classification tasks.")

    @property
    def is_classification_model(self) -> bool:
        """
        Returns whether the model is a classification model
        """
        return self.loss_type.is_classification_loss()

    @property
    def is_regression_model(self) -> bool:
        """
        Returns whether the model is a regression model
        """
        return self.loss_type.is_regression_loss()

    @property
    def is_non_imaging_model(self) -> bool:
        """
        Returns whether the model uses non image features only
        """
        return len(self.image_channels) == 0

    def should_generate_multilabel_report(self) -> bool:
        """Determines whether to produce a multilabel report. Override this to implement custom behaviour."""
        return len(self.class_names) > 1

    def get_total_number_of_non_imaging_features(self) -> int:
        """Returns the total number of non imaging features expected in the input"""
        return self.get_total_number_of_numerical_non_imaging_features() + \
               self.get_total_number_of_categorical_non_imaging_features()

    def get_total_number_of_numerical_non_imaging_features(self) -> int:
        """Returns the total number of numerical non imaging features expected in the input"""
        if len(self.numerical_columns) == 0:
            return 0
        else:
            features_channels_dict = self.get_non_image_feature_channels_dict()
            return sum([len(features_channels_dict[col]) for col in self.numerical_columns])

    def get_total_number_of_categorical_non_imaging_features(self) -> int:
        """
        Returns the total number of categorical non imaging features expected in the input eg for the
        categorical channels A and B the total number would be: 2 ( feature channels A and B) * 4
        (which is the number of bits required to one-hot encode a single channel)
        A| True, No => [1, 0, 0, 1]
        B| False, Yes => [0, 1, 1, 0]
        """
        if self.categorical_columns and not self.categorical_feature_encoder:
            raise ValueError(f"Found {len(self.categorical_columns)} categorical columns, but "
                             f"one_hot_encoder is None. Either set one_hot_encoder explicitly "
                             f"or make sure property is accessed after the dataset dataframe has been loaded.")
        elif not self.categorical_feature_encoder:
            return 0
        else:
            features_channels_dict = self.get_non_image_feature_channels_dict()
            if self.categorical_columns is None:
                return 0
            return sum([len(features_channels_dict[col]) * self.categorical_feature_encoder.get_feature_length(col)
                        for col in self.categorical_columns])

    def get_non_image_feature_channels_dict(self) -> Dict:
        """
        Convert the provided non_image_features_channels from List to Dictionary mapping each feature to its channels.
        As well as converting default key to each not defined features. Making it a property to avoid doing this
        conversion
        several time throughout the code.
        """
        if not self.non_image_feature_channels:
            return {}

        if isinstance(self.non_image_feature_channels, List):
            non_image_feature_channels_dict = {DEFAULT_KEY: self.non_image_feature_channels}
        else:
            non_image_feature_channels_dict = self.non_image_feature_channels.copy()
        all_non_image_features = self.numerical_columns.copy()
        if self.categorical_columns:
            all_non_image_features.extend(self.categorical_columns)

        # Map each feature to its channels
        for column in all_non_image_features:
            if column not in self.non_image_feature_channels:
                try:
                    non_image_feature_channels_dict[column] = non_image_feature_channels_dict[DEFAULT_KEY]
                except KeyError:
                    raise KeyError(f"The column {column} is not present in the non_image_features dictionary and the"
                                   f"default key {DEFAULT_KEY} is missing.")
        # Delete default key
        non_image_feature_channels_dict.pop(DEFAULT_KEY, None)
        return non_image_feature_channels_dict

    def filter_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframes based on expected values on columns
        :param df: the input dataframe
        :return: the filtered dataframe
        """

        def _dataframe_stats(df: pd.DataFrame) -> str:
            """
            Creates a human readable string that contains the number of rows and the number of unique subjects.
            :return: A string like "12 rows, 5 unique subjects. "
            """
            total_rows = len(df)
            if self.subject_column in df:
                unique_subjects = len(df[self.subject_column].unique())
                message = f"{unique_subjects} unique subjects"
            else:
                message = f"subject column '{self.subject_column}' not present"
            return f"{total_rows} rows, {message}. "

        logging.info(f"Before filtering: {_dataframe_stats(df)}")
        if self.expected_column_values is not None:
            for column_name, expected_value in self.expected_column_values:
                df = df[df[column_name] == expected_value]
                logging.info(f"After filtering for 'column[{column_name}] == {expected_value}': {_dataframe_stats(df)}")
        logging.info(f"Final: {_dataframe_stats(df)}")
        return df

    def get_label_transform(self) -> Union[Callable, List[Callable]]:
        """Return a transformation or list of transformation
        to apply to the labels.
        """
        return LabelTransformation.identity

    def read_dataset_into_dataframe_and_pre_process(self) -> None:
        assert self.local_dataset is not None
        file_path = self.local_dataset / self.dataset_csv
        self.dataset_data_frame = pd.read_csv(file_path, dtype=str, low_memory=False)
        self.pre_process_dataset_dataframe()

    def pre_process_dataset_dataframe(self) -> None:
        # some empty values on numeric columns get converted to nan but we want ''
        assert self.dataset_data_frame is not None
        df = self.dataset_data_frame.fillna('')
        self.dataset_data_frame = self.filter_dataframe(df)
        # update the one-hot encoder based on this dataframe
        if self.categorical_columns:
            from InnerEye.ML.utils.dataset_util import CategoricalToOneHotEncoder
            self.categorical_feature_encoder = CategoricalToOneHotEncoder.create_from_dataframe(
                dataframe=self.dataset_data_frame,
                columns=self.categorical_columns
            )

    def create_torch_datasets(self, dataset_splits: DatasetSplits) -> Dict[ModelExecutionMode, Any]:
        from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
        sample_transform = self.get_scalar_item_transform()
        assert sample_transform.train is not None  # for mypy
        assert sample_transform.val is not None  # for mypy
        assert sample_transform.test is not None  # for mypy
        train = ScalarDataset(
            args=self,
            data_frame=dataset_splits.train,
            name="training",
            sample_transform=sample_transform.train)
        val = ScalarDataset(
            args=self,
            data_frame=dataset_splits.val,
            feature_statistics=train.feature_statistics,
            name="validation",
            sample_transform=sample_transform.val)
        test = ScalarDataset(
            args=self,
            data_frame=dataset_splits.test,
            feature_statistics=train.feature_statistics,
            name="test",
            sample_transform=sample_transform.test)

        return {
            ModelExecutionMode.TRAIN: train,
            ModelExecutionMode.VAL: val,
            ModelExecutionMode.TEST: test
        }

    def create_and_set_torch_datasets(self, for_training: bool = True, for_inference: bool = True) -> None:
        """
        Creates and sets torch datasets for all model execution modes, and stores them in the self._datasets field.
        It also calls the hook to compute statistics for the train/val/test datasets.
        """
        # For models other than segmentation models, it is easier to create both training and inference datasets
        # in one go, ignoring the arguments.
        if self._datasets_for_training is None and self._datasets_for_inference is None:
            datasets = self.create_torch_datasets(self.get_dataset_splits())
            self._datasets_for_training = {mode: datasets[mode]
                                           for mode in [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]}
            self._datasets_for_inference = datasets
            for split, dataset in datasets.items():
                logging.info(f"{split.value}: {len(dataset)} subjects. Detailed status: {dataset.status}")
            if self.dataset_stats_hook:
                try:
                    self.dataset_stats_hook(datasets)
                except Exception as ex:
                    print_exception(ex, message="Error while calling the hook for computing dataset statistics.")

    def get_training_class_counts(self) -> Dict:
        if self._datasets_for_training is None:
            self.create_and_set_torch_datasets(for_inference=False)
        assert self._datasets_for_training is not None  # for mypy
        return self._datasets_for_training[ModelExecutionMode.TRAIN].get_class_counts()

    def get_total_number_of_training_samples(self) -> int:
        if self._datasets_for_training is None:
            self.create_and_set_torch_datasets(for_inference=False)
        assert self._datasets_for_training is not None  # for mypy
        return len(self._datasets_for_training[ModelExecutionMode.TRAIN])

    def create_model(self) -> Any:
        pass

    def get_loss_function(self) -> Callable:
        """Returns a custom loss function to be used with ScalarLoss.CustomClassification or CustomRegression."""
        assert self.loss_type in {ScalarLoss.CustomClassification, ScalarLoss.CustomRegression}, \
            f"get_loss_function() should be called only for custom loss types (received {self.loss_type})"
        raise NotImplementedError(f"get_loss_function() must be implemented for loss type {self.loss_type}")

    def get_post_loss_logits_normalization_function(self) -> Callable:
        """
        Post loss normalization function to apply to the logits produced by the model.
        :return:
        """
        import torch
        if self.loss_type.is_classification_loss():
            return torch.nn.Sigmoid()
        elif self.loss_type.is_regression_loss():
            return torch.nn.Identity()  # type: ignore
        else:
            raise NotImplementedError("get_post_loss_logits_normalization_function not implemented for "
                                      f"loss type: {self.loss_type}")

    def get_parameter_search_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        return super().get_parameter_search_hyperdrive_config(run_config)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return super().get_model_train_test_dataset_splits(dataset_df)

    def get_image_transform(self) -> ModelTransformsPerExecutionMode:
        """
        Get transforms to apply to images for each model execution mode.
        By default only no transformation is performed.
        """
        return ModelTransformsPerExecutionMode()

    def get_segmentation_transform(self) -> ModelTransformsPerExecutionMode:
        """
        Get transforms to apply on segmentations maps inputs for each model execution mode.
        By default only no transformation is performed.
        """
        return ModelTransformsPerExecutionMode()

    def get_scalar_item_transform(self) -> ModelTransformsPerExecutionMode:
        from InnerEye.ML.dataset.scalar_dataset import ScalarItemAugmentation
        image_transform = self.get_image_transform()
        segmentation_transform = self.get_segmentation_transform()
        return ModelTransformsPerExecutionMode(
            train=ScalarItemAugmentation(image_transform.train, segmentation_transform.train),
            val=ScalarItemAugmentation(image_transform.val, segmentation_transform.val),
            test=ScalarItemAugmentation(image_transform.test, segmentation_transform.test))

    def create_metric_computers(self) -> ModuleDict:
        """
        Gets a set of objects that compute all the metrics for the type of model that is being trained,
        across all prediction targets (sequence positions when using a sequence model).
        :return: A dictionary mapping from names of prediction targets to a list of metric computers.
        """
        # The metric computers should be stored in an object that derives from torch.Module,
        # so that they are picked up when moving the whole LightningModule to GPU.
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4713
        return ModuleDict({p: self._get_metrics_computers() for p in self.target_names})

    def _get_metrics_computers(self) -> ModuleList:
        """
        Gets the objects that compute metrics for the present kind of models, for a single prediction target.
        """
        if self.is_classification_model:
            return ModuleList([Accuracy05(),
                               AccuracyAtOptimalThreshold(),
                               OptimalThreshold(),
                               FalsePositiveRateOptimalThreshold(),
                               FalseNegativeRateOptimalThreshold(),
                               AreaUnderRocCurve(),
                               AreaUnderPrecisionRecallCurve(),
                               BinaryCrossEntropyWithLogits()])
        else:
            return ModuleList([MeanAbsoluteError(), MeanSquaredError(), ExplainedVariance()])

    def compute_and_log_metrics(self,
                                logits: torch.Tensor,
                                targets: torch.Tensor,
                                subject_ids: List[str],
                                is_training: bool,
                                metrics: ModuleDict,
                                logger: DataframeLogger,
                                current_epoch: int) -> None:
        """
        Computes all the metrics for a given (logits, labels) pair, and writes them to the loggers.
        :param logits: The model output before normalization.
        :param targets: The expected model outputs.
        :param subject_ids: The subject IDs for the present minibatch.
        :param is_training: If True, write the metrics as training metrics, otherwise as validation metrics.
        :return:
        """
        per_subject_outputs: List[Tuple[str, str, torch.Tensor, torch.Tensor]] = []
        for i, (prediction_target, metric_list) in enumerate(metrics.items()):
            # mask the model outputs and labels if required
            masked = get_masked_model_outputs_and_labels(
                logits[:, i, ...], targets[:, i, ...], subject_ids)
            # compute metrics on valid masked tensors only
            if masked is not None:
                _logits = masked.model_outputs.data
                _posteriors = self.get_post_loss_logits_normalization_function()(_logits)
                # Classification metrics expect labels as integers, but they are float throughout the rest of the code
                labels_dtype = torch.int if self.is_classification_model else _posteriors.dtype
                _labels = masked.labels.data.to(dtype=labels_dtype)
                _subject_ids = masked.subject_ids
                assert _subject_ids is not None
                for metric in metric_list:
                    if isinstance(metric, ScalarMetricsBase) and metric.compute_from_logits:
                        metric(_logits, _labels)
                    else:
                        metric(_posteriors, _labels)
                per_subject_outputs.extend(
                    zip(_subject_ids, [prediction_target] * len(_subject_ids), _posteriors.tolist(), _labels.tolist()))
        # Write a full breakdown of per-subject predictions and labels to a file. These files are local to the current
        # rank in distributed training, and will be aggregated after training.
        data_split = ModelExecutionMode.TRAIN if is_training else ModelExecutionMode.VAL
        for subject, prediction_target, model_output, label in per_subject_outputs:
            logger.add_record({
                LoggingColumns.Epoch.value: current_epoch,
                LoggingColumns.Patient.value: subject,
                LoggingColumns.Hue.value: prediction_target,
                LoggingColumns.ModelOutput.value: model_output,
                LoggingColumns.Label.value: label,
                LoggingColumns.DataSplit.value: data_split.value
            })

def get_non_image_features_dict(default_channels: List[str],
                                specific_channels: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
    """
    Returns the channels dictionary for non-imaging features.

    :param default_channels: the channels to use for all features except the features specified
    in specific_channels
    :param specific_channels: a dictionary mapping feature names to channels for all features that do
    not use the default channels
    """
    non_imaging_features_dict = {DEFAULT_KEY: default_channels}
    if specific_channels is not None:
        non_imaging_features_dict.update(specific_channels)
    return non_imaging_features_dict

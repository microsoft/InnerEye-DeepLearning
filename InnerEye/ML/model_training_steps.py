#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np
import param
import torch.cuda
import torch.utils.data
from torch.nn import MSELoss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from InnerEye.Common import common_util
from InnerEye.Common.common_util import MetricsDataframeLoggers
from InnerEye.Common.metrics_dict import MetricType, MetricsDict, create_metrics_dict_from_config
from InnerEye.Common.type_annotations import T
from InnerEye.ML import metrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import BACKGROUND_CLASS_NAME, SegmentationLoss, SegmentationModelBase
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.metrics import AzureAndTensorboardLogger, AzureMLLogger, compute_scalar_metrics
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.losses.cross_entropy import CrossEntropyLoss
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.models.losses.soft_dice import SoftDiceLoss
from InnerEye.ML.models.parallel.data_parallel import DataParallelCriterion, DataParallelModel
from InnerEye.ML.pipelines.forward_pass import SegmentationForwardPass, single_optimizer_step
from InnerEye.ML.scalar_config import AggregationType, ScalarLoss, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import dataset_util, metrics_util
from InnerEye.ML.utils.dataset_util import DatasetExample
from InnerEye.ML.utils.image_util import NumpyOrTorch
from InnerEye.ML.utils.metrics_util import SummaryWriters
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels
from InnerEye.ML.utils.supervised_criterion import BinaryCrossEntropyWithLogitsLoss, SupervisedLearningCriterion
from InnerEye.ML.visualizers.grad_cam_hooks import VisualizationMaps
from InnerEye.ML.visualizers.regression_visualization import plot_variation_error_prediction

C = TypeVar('C', bound=DeepLearningConfig)
M = TypeVar('M', bound=DeviceAwareModule)
E = TypeVar('E', List[ClassificationItemSequence[ScalarItem]], ScalarItem)


class TrainValidateParameters(param.Parameterized, Generic[M]):
    """
    Bundles parameters needed for training and validation.
      model: the model to be used
      data_loader: data loader for a cropped sample
      epoch: current epoch number
      optimizer: optimizer
    """
    model: M = param.ClassSelector(class_=DeviceAwareModule, instantiate=False)
    mean_teacher_model: M = param.ClassSelector(class_=DeviceAwareModule, instantiate=False, allow_None=True)
    data_loader: DataLoader = param.ClassSelector(class_=DataLoader, instantiate=False)
    epoch: int = param.Integer(None, bounds=(0, None))
    optimizer: Optimizer = param.ClassSelector(class_=Optimizer, instantiate=False)
    epoch_learning_rate: List[float] = param.List(None, class_=float, bounds=(1, None), instantiate=False)
    summary_writers: SummaryWriters = param.ClassSelector(class_=SummaryWriters, instantiate=False)
    in_training_mode: bool = param.Boolean(default=True)
    dataframe_loggers: MetricsDataframeLoggers = param.ClassSelector(class_=MetricsDataframeLoggers, instantiate=False)


class ModelTrainingStepsBase(Generic[C, M], ABC):
    """
    A base class that contains methods that each type of model (segmentation, classification) must implement,
    so that it can fit into the generic training routine. An implementation of this base class must have a means of
    keeping track of the training results on individual minibatches, and return them at the end of an epoch.
    """

    def __init__(self, model_config: C, train_val_params: TrainValidateParameters[M]):
        self.model_config = model_config
        self.train_val_params = train_val_params
        self.criterion = self.create_criterion()
        if self.in_training_mode:
            self.df_logger = self.train_val_params.dataframe_loggers.train_epoch_metrics
            tensorboard_logger = self.train_val_params.summary_writers.train
            azureml_logging_prefix = f"{ModelExecutionMode.TRAIN.value}_"
        else:
            self.df_logger = self.train_val_params.dataframe_loggers.val_epoch_metrics
            tensorboard_logger = self.train_val_params.summary_writers.val
            azureml_logging_prefix = f"{ModelExecutionMode.VAL.value}_"
        azureml_logger = AzureMLLogger(logging_prefix=azureml_logging_prefix,
                                       log_to_parent_run=model_config.log_to_parent_run,
                                       cross_validation_split_index=model_config.cross_validation_split_index)
        self.azure_and_tensorboard_logger = AzureAndTensorboardLogger(azureml_logger=azureml_logger,
                                                                      tensorboard_logger=tensorboard_logger,
                                                                      epoch=self.train_val_params.epoch)

    @property
    def in_training_mode(self) -> bool:
        """
        Returns True if the parameters indicate that the model should run in training mode (backpropagating the
        loss and adjusting weights). Returns False if the model should make predictions on the validation set.
        """
        return self.train_val_params.in_training_mode

    @abstractmethod
    def forward_and_backward_minibatch(self, sample: Dict[str, Any], batch_index: int, epoch: int) -> float:
        """
        Runs training for a single minibatch of training data, and returns the loss.
        :param sample: The batched sample on which the model should be trained.
        :param batch_index: The index of the present batch (supplied only for diagnostics).
        :param epoch: The number of the present epoch.
        """
        raise NotImplementedError("forward_minibatch must be implemented by derived class.")

    @abstractmethod
    def get_epoch_results_and_store(self, epoch_time_seconds: float) -> MetricsDict:
        """
        This method should assemble all training results that were achieved over all minibatches, store
        or log them in a suitable way, and then return them.
        :param epoch_time_seconds: For diagnostics, this is the total time in seconds for training the present epoch.
        :return: An object that holds an aggregate of the training results over the epoch.
        """
        raise NotImplementedError("get_epoch_results_and_store must be implemented by children")

    @abstractmethod
    def create_loss_function(self) -> torch.nn.Module:
        """
        Returns a torch module that computes a loss function.
        """
        raise NotImplementedError("create_loss_function must be implemented by children")

    def create_criterion(self) -> torch.nn.Module:
        """
        Returns a torch module that creates a criterion module which can be a DataParallelCriterion
        if use_data_parallel is enabled or the loss function module otherwise.
        """
        loss_function = self.create_loss_function()
        if self.model_config.use_data_parallel:
            return DataParallelCriterion(loss_function, self.model_config.get_cuda_devices())
        else:
            return loss_function

    def compute_loss(self, model_output: torch.Tensor, labels: NumpyOrTorch) -> torch.Tensor:
        """
        Provided model outputs (logits) applies the criterion function and returns the loss tensor.
        If data parallel is used, then the independent loss values are aggregated by averaging.
        :param model_output: Model output logits (unnormalised)
        :param labels: A tensor or numpy array of labels.
        """
        # ensure that the labels are loaded into the GPU
        labels = self.model_config.get_gpu_tensor_if_possible(labels)
        loss = self.forward_criterion(model_output, labels)
        if self.model_config.use_data_parallel:
            # Aggregate the loss values for each parallelized batch element.
            loss = torch.mean(loss)
        return loss

    def forward_criterion(self, model_output: Union[torch.Tensor, List[torch.Tensor]],
                          labels: NumpyOrTorch) -> torch.Tensor:
        """
        Handles the forward pass for the loss function.
        :param model_output: A single Tensor, or a list if using DataParallelCriterion
        :param labels: Labels to compute loss against.
        :return: loss tensor.
        """
        return self.criterion(model_output, labels)


@dataclass
class ScalarModelInputsAndLabels(Generic[E, T]):
    """
    Holds the results of calling get_scalar_model_inputs_and_labels: For a given sample returned by the data loader,
    create the model inputs, the labels, the list of subjects (data loader sample can be batched),
    and the reconstructed data item.
    """
    model_inputs: List[torch.Tensor]
    labels: T
    subject_ids: List[str]
    data_item: E

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


def get_scalar_model_inputs_and_labels(model_config: ScalarModelBase,
                                       model: torch.nn.Module,
                                       sample: Dict[str, Any]) -> ScalarModelInputsAndLabels:
    """
    For a model that predicts scalars, gets the model input tensors from a sample returned by the data loader.
    :param model_config: The configuration object for the model.
    :param model: The instantiated PyTorch model.
    :param sample: A training sample, as returned by a PyTorch data loader (dictionary mapping from field name to value)
    :return: An instance of ScalarModelInputsAndLabels, containing the list of model input tensors,
    label tensor, subject IDs, and the data item reconstructed from the data loader output
    """
    if isinstance(model, DataParallelModel):
        model = model.get_module()

    if isinstance(model_config, SequenceModelBase):
        sequence_model: DeviceAwareModule[List[ClassificationItemSequence], torch.Tensor] = model
        sequences = ClassificationItemSequence.from_minibatch(sample)
        subject_ids = [x.id for x in sequences]
        labels = ClassificationItemSequence.create_labels_tensor_for_minibatch(
            sequences=sequences,
            target_indices=model_config.get_target_indices()
        )
        model_inputs = sequence_model.get_input_tensors(sequences)

        return ScalarModelInputsAndLabels[List[ClassificationItemSequence], torch.Tensor](
            model_inputs=model_inputs,
            labels=labels,
            subject_ids=subject_ids,
            data_item=sequences
        )
    else:
        scalar_model: DeviceAwareModule[ScalarItem, torch.Tensor] = model
        scalar_item = ScalarItem.from_dict(sample)
        subject_ids = [str(x.id) for x in scalar_item.metadata]  # type: ignore
        model_inputs = scalar_model.get_input_tensors(scalar_item)

        return ScalarModelInputsAndLabels[ScalarItem, torch.Tensor](
            model_inputs=model_inputs,
            labels=scalar_item.label,
            subject_ids=subject_ids,
            data_item=scalar_item
        )


F = TypeVar("F", bound=ScalarModelBase)


class ModelTrainingStepsForScalarModel(ModelTrainingStepsBase[F, DeviceAwareModule]):
    """
    This class implements all steps necessary for training an image classification model during a single epoch.
    """

    def __init__(self, config: F, train_val_params: TrainValidateParameters[DeviceAwareModule]):
        """
        Creates a new instance of the class.
        :param config: The configuration of a classification model.
        :param train_val_params: The parameters for training the model, including the optimizer and the data loaders.
        """
        # This field needs to be defined in the constructor to keep pycharm happy, but before the call to the
        # base class because the base class constructor create_loss_function
        self.label_tensor_dtype = torch.float32
        super().__init__(config, train_val_params)
        self.metrics = create_metrics_dict_from_config(config)
        self.compute_mean_teacher_model = self.model_config.compute_mean_teacher_model
        if self.model_config.compute_grad_cam:
            if not self.model_config.aggregation_type == AggregationType.Average:
                self.model_config.max_batch_grad_cam = 0
                logging.warning("GradCam computation is not implemented for this aggregation type."
                                "Ignoring computation.")
            else:
                model_to_evaluate = self.train_val_params.mean_teacher_model if \
                    self.model_config.compute_mean_teacher_model else self.train_val_params.model
                self.guided_grad_cam = VisualizationMaps(model_to_evaluate, self.model_config)
                self.model_config.visualization_folder.mkdir(exist_ok=True)

    def create_loss_function(self) -> torch.nn.Module:
        """
        Returns a torch module that computes a loss function.
        Depending on the chosen loss function, the required data type for the labels tensor is set in
        self.
        """
        if self.model_config.loss_type == ScalarLoss.BinaryCrossEntropyWithLogits:
            return BinaryCrossEntropyWithLogitsLoss(smoothing_eps=self.model_config.label_smoothing_eps)
        if self.model_config.loss_type == ScalarLoss.WeightedCrossEntropyWithLogits:
            return BinaryCrossEntropyWithLogitsLoss(
                smoothing_eps=self.model_config.label_smoothing_eps,
                class_counts=self.model_config.get_training_class_counts())
        elif self.model_config.loss_type == ScalarLoss.MeanSquaredError:
            self.label_tensor_dtype = torch.float32
            return MSELoss()
        else:
            raise NotImplementedError("Loss type {} is not implemented".format(self.model_config.loss_type))

    def get_label_tensor(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Converts the given tensor to the right data format, depending on the chosen loss function.
        :param labels: The label tensor that should be converted.
        """
        try:
            labels = labels.to(dtype=self.label_tensor_dtype)
        except ValueError as ex:
            raise ValueError(f"Unable to convert tensor {labels} to data type {self.label_tensor_dtype}: {str(ex)}")
        return self.model_config.get_gpu_tensor_if_possible(labels)

    def get_logits_and_outputs(self, *model_inputs: torch.Tensor, use_mean_teacher_model: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a Tuple containing the logits and the final model output. Note that the logits might be
        distributed over multiple GPU if the model is an instance of DataParallel. In this case, the model outputs on
        the other hand will be gathered to GPU_0.

        :param model_inputs: input to evaluate the model on
        :param use_mean_teacher_model: If True, logits and outputs are produced for the mean teacher model. Else
        logits and outputs are produced for the standard (student) model.
        :return: Tuple (logits, model_output).
        """
        if use_mean_teacher_model:
            logits = self.train_val_params.mean_teacher_model(*model_inputs)
        else:
            logits = self.train_val_params.model(*model_inputs)
        if isinstance(logits, list):
            # When using multiple GPUs, logits is a list of tensors. Gather will concatenate them
            # across the first dimension, and move them to GPU0.
            model_output = torch.nn.parallel.gather(logits, target_device=0)
        else:
            model_output = logits
        model_output = self.model_config.get_post_loss_logits_normalization_function()(model_output)
        return logits, model_output

    def forward_and_backward_minibatch(self, sample: Dict[str, Any], batch_index: int, epoch: int) -> float:
        """
        Runs training for a single minibatch of training data, and computes all metrics.
        :param sample: The batched sample on which the model should be trained.
        :param batch_index: The index of the present batch (supplied only for diagnostics).
        :param epoch: The number of the present epoch.
        """
        start_time = time.time()
        model = self.train_val_params.model
        mean_teacher_model = self.train_val_params.mean_teacher_model
        model_inputs_and_labels = get_scalar_model_inputs_and_labels(self.model_config, model, sample)

        if self.in_training_mode:
            model.train()
            logits, model_output = self.get_logits_and_outputs(*model_inputs_and_labels.model_inputs)
        else:
            model.eval()
            with torch.no_grad():
                logits, model_output = self.get_logits_and_outputs(*model_inputs_and_labels.model_inputs)
            model.train()

        label_gpu = self.get_label_tensor(model_inputs_and_labels.labels)
        loss = self.compute_loss(logits, label_gpu)

        if self.in_training_mode:
            single_optimizer_step(self.model_config, loss, self.train_val_params.optimizer)
            if self.model_config.compute_mean_teacher_model:
                self.update_mean_teacher_parameters()

        if self.compute_mean_teacher_model:
            # If the mean teacher model is computed, use the output of the mean teacher for the metrics report
            # instead of the output of the student model.
            mean_teacher_model.eval()
            with torch.no_grad():
                _, model_output = self.get_logits_and_outputs(*model_inputs_and_labels.model_inputs,
                                                              use_mean_teacher_model=True)

        if self._should_save_grad_cam_output(epoch=epoch, batch_index=batch_index):
            self.save_grad_cam(epoch, model_inputs_and_labels.subject_ids,
                               model_inputs_and_labels.data_item,
                               model_inputs_and_labels.model_inputs,
                               label_gpu)

        self.metrics.add_metric(MetricType.LOSS, loss.item())
        self.update_metrics(model_inputs_and_labels.subject_ids, model_output, label_gpu)
        logging.debug(f"Batch {batch_index}: {self.metrics.to_string()}")
        minibatch_time = time.time() - start_time
        self.metrics.add_metric(MetricType.SECONDS_PER_BATCH, minibatch_time)
        return loss.item()

    def get_epoch_results_and_store(self, epoch_time_seconds: float) -> MetricsDict:
        """
        Assembles all training results that were achieved over all minibatches, returns them as a dictionary
        mapping from metric name to metric value.
        :param epoch_time_seconds: For diagnostics, this is the total time in seconds for training the present epoch.
        :return: A dictionary that holds all metrics averaged over the epoch.
        """
        self.metrics.add_metric(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds)
        assert len(self.train_val_params.epoch_learning_rate) == 1, "Expected a single entry for learning rate."
        self.metrics.add_metric(MetricType.LEARNING_RATE, self.train_val_params.epoch_learning_rate[0])
        averaged_across_hues = self.metrics.average(across_hues=False)
        mode = ModelExecutionMode.TRAIN if self.in_training_mode else ModelExecutionMode.VAL
        diagnostics_lines = averaged_across_hues.to_string()
        logging.info(f"Results for epoch {self.train_val_params.epoch:3d} {mode.value}\n{diagnostics_lines}")

        # Store subject level metrics
        subject_logger = self.train_val_params.dataframe_loggers.train_subject_metrics if \
            self.train_val_params.in_training_mode \
            else self.train_val_params.dataframe_loggers.val_subject_metrics
        self.metrics.store_metrics_per_subject(
            epoch=self.train_val_params.epoch,
            df_logger=subject_logger,
            mode=mode,
            cross_validation_split_index=self.model_config.cross_validation_split_index)

        if self._should_save_regression_error_plot(self.train_val_params.epoch):
            error_plot_name = f"error_plot_{self.train_val_params.epoch}"
            path = str(self.model_config.outputs_folder / f"{error_plot_name}.png")
            plot_variation_error_prediction(self.metrics.get_labels(), self.metrics.get_predictions(), path)
            self.azure_and_tensorboard_logger.log_image(error_plot_name, path)

        # Write metrics to Azure and TensorBoard
        metrics.store_epoch_metrics(self.azure_and_tensorboard_logger,
                                    self.df_logger,
                                    self.train_val_params.epoch,
                                    averaged_across_hues,
                                    self.train_val_params.epoch_learning_rate,
                                    self.model_config)
        return self.metrics.average(across_hues=True)

    def update_metrics(self, subject_ids: List[str], model_output: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Handle metrics updates based on the provided model outputs and labels.
        """
        compute_scalar_metrics(self.metrics, subject_ids, model_output, labels, self.model_config.loss_type)

    def save_grad_cam(self,
                      epoch: int,
                      subject_ids: List,
                      classification_item: Union[List[ClassificationItemSequence[ScalarItem]], ScalarItem],
                      model_inputs: List[torch.Tensor],
                      labels: torch.Tensor) -> None:
        filenames = [f"{epoch}_viz_{id}" for id in subject_ids]
        self.guided_grad_cam.save_visualizations_in_notebook(
            classification_item,  # type: ignore
            model_inputs,
            filenames,
            ground_truth_labels=labels.cpu().numpy(),
            gradcam_dir=self.model_config.visualization_folder
        )

    def _should_save_grad_cam_output(self, epoch: int, batch_index: int) -> bool:
        return self.model_config.is_classification_model \
               and (not self.in_training_mode) \
               and self.model_config.should_save_epoch(epoch) \
               and (batch_index < self.model_config.max_batch_grad_cam)

    def _should_save_regression_error_plot(self, epoch: int) -> bool:
        return self.model_config.is_regression_model \
               and (not self.in_training_mode) \
               and self.model_config.should_save_epoch(epoch)

    def update_mean_teacher_parameters(self) -> None:
        """
        Updates the mean teacher model parameters as per the update formula
        mean_teacher_model_weight = alpha * (mean_teacher_model_weight) + (1-alpha) * (student_model_weight)
        see https://arxiv.org/abs/1703.01780
        """
        mean_teacher_model = self.train_val_params.mean_teacher_model
        model = self.train_val_params.model
        if isinstance(mean_teacher_model, DataParallelModel):
            mean_teacher_model = mean_teacher_model.module  # type: ignore
            model = model.module  # type: ignore
        for mean_teacher_param, student_param in zip(mean_teacher_model.parameters(), model.parameters()):
            mean_teacher_param.data = self.model_config.mean_teacher_alpha * mean_teacher_param.data \
                                      + (1 - self.model_config.mean_teacher_alpha) * student_param.data


class ModelTrainingStepsForSequenceModel(ModelTrainingStepsForScalarModel[SequenceModelBase]):
    """
    This class implements all steps necessary for training an sequence model during a single epoch.
    """

    def forward_criterion(self, model_output: Union[torch.Tensor, List[torch.Tensor]],
                          labels: NumpyOrTorch) -> torch.Tensor:
        _model_output: torch.Tensor
        # we need to gather the model outputs before masking them for the criterion.
        if isinstance(model_output, list):
            # When using multiple GPUs, model_output is a list of tensors. Gather will concatenate them
            # across the first dimension, and move them to GPU0.
            _model_output = torch.nn.parallel.gather(model_output, target_device=0)
        else:
            _model_output = model_output

        # create masked sequences based on the labels
        masked_model_outputs_and_labels = get_masked_model_outputs_and_labels(_model_output, labels)
        if masked_model_outputs_and_labels is None:
            raise ValueError("Invalid model_output and labels found")

        # do not use a data parallel criterion as we have gathered the model outputs
        if isinstance(self.criterion, DataParallelCriterion):
            criterion = self.criterion.module  # type: ignore
        else:
            criterion = self.criterion

        return criterion(masked_model_outputs_and_labels.model_outputs, masked_model_outputs_and_labels.labels)


class ModelTrainingStepsForSegmentation(ModelTrainingStepsBase[SegmentationModelBase, DeviceAwareModule]):
    """
    This class implements all steps necessary for training an image segmentation model during a single epoch.
    """

    def __init__(self, model_config: SegmentationModelBase,
                 train_val_params: TrainValidateParameters[DeviceAwareModule]):
        """
        Creates a new instance of the class.
        :param model_config: The configuration of a segmentation model.
        :param train_val_params: The parameters for training the model, including the optimizer and the data loaders.
        """
        super().__init__(model_config, train_val_params)
        self.example_to_save = np.random.randint(0, len(train_val_params.data_loader))
        self.pipeline = SegmentationForwardPass(model=self.train_val_params.model,
                                                model_config=self.model_config,
                                                batch_size=self.model_config.train_batch_size,
                                                optimizer=self.train_val_params.optimizer,
                                                in_training_mode=self.train_val_params.in_training_mode,
                                                criterion=self.compute_loss)
        self.metrics = MetricsDict(hues=[BACKGROUND_CLASS_NAME] + model_config.ground_truth_ids)

    def create_loss_function(self) -> torch.nn.Module:
        """
        Returns a torch module that computes a loss function.
        """
        return self.construct_loss_function(self.model_config)

    @classmethod
    def construct_loss_function(cls, model_config: SegmentationModelBase) -> SupervisedLearningCriterion:
        """
        Returns a loss function from the model config; mixture losses are constructed as weighted combinations of
        other loss functions.
        """
        if model_config.loss_type == SegmentationLoss.Mixture:
            components = model_config.mixture_loss_components
            assert components is not None
            sum_weights = sum(component.weight for component in components)
            weights_and_losses = []
            for component in components:
                normalized_weight = component.weight / sum_weights
                loss_function = cls.construct_non_mixture_loss_function(model_config, component.loss_type,
                                                                        component.class_weight_power)
                weights_and_losses.append((normalized_weight, loss_function))
            return MixtureLoss(weights_and_losses)
        return cls.construct_non_mixture_loss_function(model_config, model_config.loss_type,
                                                       model_config.loss_class_weight_power)

    @classmethod
    def construct_non_mixture_loss_function(cls,
                                            model_config: SegmentationModelBase,
                                            loss_type: SegmentationLoss,
                                            power: Optional[float]) -> SupervisedLearningCriterion:
        """
        :param model_config: model configuration to get some parameters from
        :param loss_type: type of loss function
        :param power: value for class_weight_power for the loss function
        :return: instance of loss function
        """
        if loss_type == SegmentationLoss.SoftDice:
            return SoftDiceLoss(class_weight_power=power)
        elif loss_type == SegmentationLoss.CrossEntropy:
            return CrossEntropyLoss(class_weight_power=power,
                                    smoothing_eps=model_config.label_smoothing_eps,
                                    focal_loss_gamma=None)
        elif loss_type == SegmentationLoss.Focal:
            return CrossEntropyLoss(class_weight_power=power,
                                    smoothing_eps=model_config.label_smoothing_eps,
                                    focal_loss_gamma=model_config.focal_loss_gamma)
        else:
            raise NotImplementedError("Loss type {} is not implemented".format(loss_type))

    def forward_and_backward_minibatch(self, sample: Dict[str, Any], batch_index: int, epoch: int) -> float:
        """
        Runs training for a single minibatch of training data, and computes all metrics.
        :param sample: The batched sample on which the model should be trained.
        :param batch_index: The index of the present batch (supplied only for diagnostics).
        :param epoch: The number of the present epoch.
        """
        cropped_sample: CroppedSample = CroppedSample.from_dict(sample=sample)
        labels = self.model_config.get_gpu_tensor_if_possible(cropped_sample.labels_center_crop)

        mask = None if self.train_val_params.in_training_mode else cropped_sample.mask_center_crop
        forward_pass_result = self.pipeline.forward_pass_patches(patches=cropped_sample.image,
                                                                 labels=labels,
                                                                 mask=mask)
        # Clear the GPU cache between forward and backward passes to avoid possible out-of-memory
        torch.cuda.empty_cache()
        dice_for_all_classes = metrics.compute_dice_across_patches(
            segmentation=torch.tensor(forward_pass_result.segmentations).long(),
            ground_truth=labels,
            use_cuda=self.model_config.use_gpu,
            allow_multiple_classes_for_each_pixel=True).cpu().numpy()
        foreground_voxels = metrics_util.get_number_of_voxels_per_class(cropped_sample.labels)
        # loss is a scalar, also when running the forward pass over multiple crops.
        # dice_for_all_structures has one row per crop.
        if forward_pass_result.loss is None:
            raise ValueError("During training, the loss should always be computed, but the value is None.")
        loss = forward_pass_result.loss

        # store metrics per batch
        self.metrics.add_metric(MetricType.LOSS, loss)
        for i, ground_truth_id in enumerate(self.metrics.get_hue_names(include_default=False)):
            for b in range(dice_for_all_classes.shape[0]):
                self.metrics.add_metric(MetricType.DICE, dice_for_all_classes[b, i].item(),
                                        hue=ground_truth_id, skip_nan_when_averaging=True)
            self.metrics.add_metric(MetricType.VOXEL_COUNT, foreground_voxels[i], hue=ground_truth_id)
        # store diagnostics per batch
        center_indices = cropped_sample.center_indices
        if isinstance(center_indices, torch.Tensor):
            center_indices = center_indices.cpu().numpy()
        self.metrics.add_diagnostics(MetricType.PATCH_CENTER.value, np.copy(center_indices))
        if self.train_val_params.in_training_mode:
            # store the sample train patch from this epoch for visualization
            if batch_index == self.example_to_save and self.model_config.store_dataset_sample:
                _store_dataset_sample(self.model_config, self.train_val_params.epoch, forward_pass_result,
                                      cropped_sample)
        return loss

    def get_epoch_results_and_store(self, epoch_time_seconds: float) -> MetricsDict:
        """
        Assembles all training results that were achieved over all minibatches, writes them to Tensorboard and
        AzureML, and returns them as a MetricsDict object.
        :param epoch_time_seconds: For diagnostics, this is the total time in seconds for training the present epoch.
        :return: A dictionary that holds all metrics averaged over the epoch.
        """
        self.metrics.add_metric(MetricType.SECONDS_PER_EPOCH, epoch_time_seconds)
        assert len(self.train_val_params.epoch_learning_rate) == 1, "Expected a single entry for learning rate."
        self.metrics.add_metric(MetricType.LEARNING_RATE, self.train_val_params.epoch_learning_rate[0])
        result = metrics.aggregate_segmentation_metrics(self.metrics)
        metrics.store_epoch_metrics(self.azure_and_tensorboard_logger,
                                    self.df_logger,
                                    self.train_val_params.epoch,
                                    result,
                                    self.train_val_params.epoch_learning_rate,
                                    self.model_config)
        return result


# noinspection PyUnresolvedReferences
def _store_dataset_sample(config: SegmentationModelBase,
                          epoch: int,
                          forward_pass_result: SegmentationForwardPass.Result,
                          sample: CroppedSample) -> None:
    """
    Stores the first sample in a batch, along with it's results from the model forward pass
    as Nifti to the file system.
    :param config: Training configurations.
    :param epoch: The epoch to which this sample belongs to.
    :param forward_pass_result: The result of a model forward pass.
    :param sample: The original crop sample used for training, as returned by the data loader
    """
    # pick the first image from the batch as example
    example = DatasetExample(epoch=epoch,
                             # noinspection PyTypeChecker
                             patient_id=sample.metadata[0].patient_id,  # type: ignore
                             image=sample.image[0][0].numpy(),
                             labels=sample.labels[0].numpy(),
                             prediction=forward_pass_result.segmentations[0],
                             header=sample.metadata[0].image_header)  # type: ignore
    dataset_util.store_and_upload_example(dataset_example=example, args=config)

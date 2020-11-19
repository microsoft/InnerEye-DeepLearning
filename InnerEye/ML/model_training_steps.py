#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union

import param
import torch.cuda
import torch.utils.data
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import MSELoss
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from InnerEye.Common.metrics_dict import MetricType, MetricsDict, create_metrics_dict_from_config
from InnerEye.ML import metrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.metrics import compute_scalar_metrics
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.models.losses.ece import ECELoss
from InnerEye.ML.models.parallel.data_parallel import DataParallelCriterion, DataParallelModel, \
    execute_within_autocast_if_needed
from InnerEye.ML.pipelines.forward_pass import SegmentationForwardPass, single_optimizer_step
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import dataset_util
from InnerEye.ML.utils.dataset_util import DatasetExample
from InnerEye.ML.utils.image_util import NumpyOrTorch
from InnerEye.ML.utils.model_util import ScalarModelInputsAndLabels, get_scalar_model_inputs_and_labels
from InnerEye.ML.utils.sequence_utils import get_masked_model_outputs_and_labels
from InnerEye.ML.utils.supervised_criterion import BinaryCrossEntropyWithLogitsLoss
from InnerEye.ML.utils.temperature_scaling import ModelWithTemperature
from InnerEye.ML.utils.training_util import ModelForwardAndBackwardsOutputs, gather_tensor
from InnerEye.ML.visualizers.grad_cam_hooks import VisualizationMaps
from InnerEye.ML.visualizers.regression_visualization import plot_variation_error_prediction

C = TypeVar('C', bound=DeepLearningConfig)
M = TypeVar('M', bound=DeviceAwareModule)


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
    in_training_mode: bool = param.Boolean(default=True)
    save_metrics: bool = param.Boolean(default=True)


class TrainingAndValidationDataForSegmentation(LightningDataModule):
    def _init__(self, config: ModelConfigBase):
        super().__init__()
        self.config = config
        self.data_loaders = None

    def setup(self, stage: Optional[str] = None) -> None:
        self.data_loaders = self.config.create_data_loaders()

    def train_dataloader(self):
        return self.data_loaders[ModelExecutionMode.TRAIN]

    def val_dataloader(self):
        return self.data_loaders[ModelExecutionMode.VAL]

    def test_dataloader(self):
        raise NotImplementedError("For segmentation models, the test dataset should not be evaluated patch-wise.")


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

    @property
    def in_training_mode(self) -> bool:
        """
        Returns True if the parameters indicate that the model should run in training mode (backpropagating the
        loss and adjusting weights). Returns False if the model should make predictions on the validation set.
        """
        return self.train_val_params.in_training_mode

    @abstractmethod
    def forward_and_backward_minibatch(self, sample: Dict[str, Any],
                                       batch_index: int, epoch: int) -> ModelForwardAndBackwardsOutputs:
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

    def compute_loss(self, model_output: torch.Tensor, labels: NumpyOrTorch) -> torch.Tensor:
        """
        Provided model outputs (logits) applies the criterion function and returns the loss tensor.
        If data parallel is used, then the independent loss values are aggregated by averaging.
        :param model_output: Model output logits (unnormalised)
        :param labels: A tensor or numpy array of labels.
        """
        # ensure that the labels are loaded into the GPU
        labels = self.model_config.get_gpu_tensor_if_possible(labels)
        loss = self.forward_criterion_with_autocast(model_output, labels)
        if self.model_config.use_data_parallel:
            # Aggregate the loss values for each parallelized batch element.
            loss = torch.mean(loss)
        return loss


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

    def get_logits_and_posteriors(self, *model_inputs: torch.Tensor, use_mean_teacher_model: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a Tuple containing the logits and the final model output. Note that the logits might be
        distributed over multiple GPU if the model is an instance of DataParallel. In this case,
        the posteriors will be gathered to GPU_0.
        :param model_inputs: input to evaluate the model on
        :param use_mean_teacher_model: If True, logits and posteriors are produced for the mean teacher model. Else
        logits and posteriors are produced for the standard (student) model.
        :return: Tuple (logits, posteriors).
        """
        if use_mean_teacher_model:
            logits = self.train_val_params.mean_teacher_model(*model_inputs)
        else:
            logits = self.train_val_params.model(*model_inputs)
        posteriors = self.model_config.get_post_loss_logits_normalization_function()(gather_tensor(logits))
        return logits, posteriors

    def _compute_model_output_and_loss(self, model_inputs_and_labels: ScalarModelInputsAndLabels) -> \
            Tuple[Tensor, Tensor, Tensor]:
        """
        Computes the output of the model for a given set of inputs and labels.
        Returns a tuple of (logits, posteriors, loss). For multi-GPU computation, the logits are returned
        as a list.
        """
        model = self.train_val_params.model
        label_gpu = self.get_label_tensor(model_inputs_and_labels.labels)
        if self.model_config.use_mixed_precision and self.model_config.use_gpu:
            label_gpu = label_gpu.to(dtype=torch.float16)

        def compute() -> Tuple[Tensor, Tensor, Tensor]:
            if self.in_training_mode:
                model.train()
                logits, posteriors = self.get_logits_and_posteriors(*model_inputs_and_labels.model_inputs)
            else:
                model.eval()
                with torch.no_grad():
                    logits, posteriors = self.get_logits_and_posteriors(*model_inputs_and_labels.model_inputs)
                model.train()
            loss = self.compute_loss(logits, label_gpu)
            return logits, posteriors, loss

        return execute_within_autocast_if_needed(func=compute, use_autocast=self.model_config.use_mixed_precision)

    def forward_and_backward_minibatch(self, sample: Dict[str, Any],
                                       batch_index: int, epoch: int) -> ModelForwardAndBackwardsOutputs:
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
        label_gpu = self.get_label_tensor(model_inputs_and_labels.labels)
        logits, posteriors, loss = self._compute_model_output_and_loss(model_inputs_and_labels)
        gathered_logits = gather_tensor(logits)
        if self.in_training_mode:
            single_optimizer_step(loss,
                                  self.train_val_params.optimizer,
                                  self.train_val_params.gradient_scaler)
            if self.model_config.compute_mean_teacher_model:
                self.update_mean_teacher_parameters()

        if self.compute_mean_teacher_model:
            # If the mean teacher model is computed, use the output of the mean teacher for the metrics report
            # instead of the output of the student model.
            mean_teacher_model.eval()
            with torch.no_grad():
                logits, posteriors = self.get_logits_and_posteriors(
                    *model_inputs_and_labels.model_inputs,
                    use_mean_teacher_model=True)
                gathered_logits = gather_tensor(logits)

        # Autocast may have returned float16 tensors. Documentation suggests to simply cast back to float32.
        # If tensor was already float32, no overhead is incurred.
        posteriors = posteriors.detach().float()
        gathered_logits = gathered_logits.detach().float().cpu()
        loss_scalar = loss.float().item()

        if self.train_val_params.save_metrics:
            if self._should_save_grad_cam_output(epoch=epoch, batch_index=batch_index):
                self.save_grad_cam(epoch, model_inputs_and_labels.subject_ids,
                                   model_inputs_and_labels.data_item,
                                   model_inputs_and_labels.model_inputs,
                                   label_gpu)

            self.metrics.add_metric(MetricType.LOSS, loss_scalar)
            self.update_metrics(model_inputs_and_labels.subject_ids, posteriors, label_gpu)
            logging.debug(f"Batch {batch_index}: {self.metrics.to_string()}")
            minibatch_time = time.time() - start_time
            self.metrics.add_metric(MetricType.SECONDS_PER_BATCH, minibatch_time)

        return ModelForwardAndBackwardsOutputs(
            loss=loss_scalar,
            logits=gathered_logits,
            labels=model_inputs_and_labels.labels
        )

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

    def _should_save_grad_cam_output(self, epoch: int, batch_index: int) -> bool:
        return self.model_config.is_classification_model \
               and (not self.in_training_mode) \
               and self.model_config.should_save_epoch(epoch) \
               and (batch_index < self.model_config.max_batch_grad_cam)

    def _should_save_regression_error_plot(self, epoch: int) -> bool:
        return self.model_config.is_regression_model \
               and (not self.in_training_mode) \
               and self.model_config.should_save_epoch(epoch)


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

    def learn_temperature_scale_parameter(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Uses the provided logits and labels to learn a temperature scale parameter.
        :param logits: Logits to use in order to learn a temperature scale parameter
        :param labels: Labels to use in order to learn a temperature scale parameter
        :return Optimal temperature value
        """
        _model: Union[DeviceAwareModule, DataParallelModel, ModelWithTemperature] = self.train_val_params.model
        assert self.model_config.temperature_scaling_config is not None
        ece_criterion: ECELoss = ECELoss(activation=self.model_config.get_post_loss_logits_normalization_function(),
                                         n_bins=self.model_config.temperature_scaling_config.ece_num_bins)

        if self.model_config.use_gpu:
            ece_criterion = ece_criterion.cuda()
        if isinstance(_model, DataParallelModel):
            _model = _model.get_module()

        def _forward_criterion(_logits: torch.Tensor, _labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            loss = self.forward_criterion_with_autocast(_logits, _labels).to(torch.float32)
            masked_model_outputs_and_labels = get_masked_model_outputs_and_labels(_logits, _labels)
            assert masked_model_outputs_and_labels is not None
            ece = ece_criterion(masked_model_outputs_and_labels.model_outputs.data.unsqueeze(dim=0),
                                masked_model_outputs_and_labels.labels.data.unsqueeze(dim=0))
            return loss, ece

        assert isinstance(_model, ModelWithTemperature)
        return _model.set_temperature(
            logits=logits,
            labels=labels,
            criterion_fn=_forward_criterion,
            use_gpu=self.model_config.use_gpu,
            logger=self.azure_and_tensorboard_logger
        )


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

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

import torch
from torch.nn import MSELoss
from torch.nn.parameter import Parameter
from torch.optim.rmsprop import RMSprop

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.Common import common_util
from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import ModelArchitectureConfig, PaddingMode, SegmentationLoss, SegmentationModelBase, \
    basic_size_shrinkage
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.deep_learning_config import DeepLearningConfig, OptimizerType
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel, CropSizeConstraints
from InnerEye.ML.models.architectures.complex import ComplexModel
from InnerEye.ML.models.architectures.unet_2d import UNet2D
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.losses.cross_entropy import CrossEntropyLoss
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.models.losses.soft_dice import SoftDiceLoss
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.ml_util import RandomStateSnapshot
from InnerEye.ML.utils.supervised_criterion import BinaryCrossEntropyWithLogitsLoss, SupervisedLearningCriterion
from InnerEye.ML.utils.temperature_scaling import ModelWithTemperature
from InnerEye.ML.visualizers.model_summary import ModelSummary


def create_optimizer(config: DeepLearningConfig, parameters: Iterator[Parameter]) -> torch.optim.Optimizer:
    # Select optimizer type
    if config.optimizer_type in [OptimizerType.Adam, OptimizerType.AMSGrad]:
        return torch.optim.Adam(parameters, config.l_rate,
                                config.adam_betas, config.opt_eps, config.weight_decay,
                                amsgrad=config.optimizer_type == OptimizerType.AMSGrad)
    elif config.optimizer_type == OptimizerType.SGD:
        return torch.optim.SGD(parameters, config.l_rate, config.momentum,
                               weight_decay=config.weight_decay)
    elif config.optimizer_type == OptimizerType.RMSprop:
        return RMSprop(parameters, config.l_rate, config.rms_alpha,
                       config.opt_eps,
                       config.weight_decay, config.momentum)
    else:
        raise NotImplementedError(f"Optimizer type {config.optimizer_type.value} is not implemented")


def create_segmentation_loss_function(model_config: SegmentationModelBase) -> SupervisedLearningCriterion:
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
            loss_function = create_segmentation_loss_component(model_config,
                                                               component.loss_type,
                                                               component.class_weight_power)
            weights_and_losses.append((normalized_weight, loss_function))
        return MixtureLoss(weights_and_losses)
    return create_segmentation_loss_component(model_config,
                                              model_config.loss_type,
                                              model_config.loss_class_weight_power)


def create_segmentation_loss_component(model_config: SegmentationModelBase,
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


def create_scalar_loss_function(config: ScalarModelBase) -> torch.nn.Module:
    """
    Returns a torch module that computes a loss function for classification and regression models.
    """
    if config.loss_type == ScalarLoss.BinaryCrossEntropyWithLogits:
        return BinaryCrossEntropyWithLogitsLoss(num_classes=len(config.class_names),
                                                smoothing_eps=config.label_smoothing_eps)
    if config.loss_type == ScalarLoss.WeightedCrossEntropyWithLogits:
        return BinaryCrossEntropyWithLogitsLoss(
            num_classes=len(config.class_names),
            smoothing_eps=config.label_smoothing_eps,
            class_counts=config.get_training_class_counts(),
            num_train_samples=config.get_total_number_of_training_samples())
    elif config.loss_type == ScalarLoss.MeanSquaredError:
        return MSELoss()
    else:
        raise NotImplementedError(f"Loss type {config.loss_type} is not implemented")


def init_weights(m: Union[torch.nn.Conv3d, torch.nn.BatchNorm3d]) -> None:
    """
    Initialize the weights of a Pytorch module.

    :param m: A PyTorch module. Only Conv3d and BatchNorm3d are initialized.
    """
    import torch
    if isinstance(m, torch.nn.Conv3d):
        torch.nn.init.normal_(m.weight, 0, 0.01)
    elif isinstance(m, torch.nn.BatchNorm3d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


# noinspection PyTypeChecker
def build_net(args: SegmentationModelBase) -> BaseSegmentationModel:
    """
    Build network architectures

    :param args: Network configuration arguments
    """
    full_channels_list = [args.number_of_image_channels, *args.feature_channels, args.number_of_classes]
    initial_fcn = [BasicLayer] * 2
    residual_blocks = [[BasicLayer, BasicLayer]] * 3
    basic_network_definition = initial_fcn + residual_blocks  # type: ignore
    # no dilation for the initial FCN and then a constant 1 neighbourhood dilation for the rest residual blocks
    basic_dilations = [1] * len(initial_fcn) + [2, 2] * len(basic_network_definition)
    # Crop size must be at least 29 because all architectures (apart from UNets) shrink the input image by 28
    crop_size_constraints = CropSizeConstraints(minimum_size=basic_size_shrinkage + 1)
    run_weight_initialization = True

    network: BaseSegmentationModel
    if args.architecture == ModelArchitectureConfig.Basic:
        network_definition = basic_network_definition
        network = ComplexModel(args, full_channels_list,
                               basic_dilations, network_definition, crop_size_constraints)  # type: ignore

    elif args.architecture == ModelArchitectureConfig.UNet3D:
        network = UNet3D(input_image_channels=args.number_of_image_channels,
                         initial_feature_channels=args.feature_channels[0],
                         num_classes=args.number_of_classes,
                         kernel_size=args.kernel_size,
                         num_downsampling_paths=args.num_downsampling_paths)
        run_weight_initialization = False

    elif args.architecture == ModelArchitectureConfig.UNet2D:
        network = UNet2D(input_image_channels=args.number_of_image_channels,
                         initial_feature_channels=args.feature_channels[0],
                         num_classes=args.number_of_classes,
                         padding_mode=PaddingMode.Edge,
                         num_downsampling_paths=args.num_downsampling_paths)
        run_weight_initialization = False

    else:
        raise ValueError(f"Unknown model architecture {args.architecture}")
    network.validate_crop_size(args.crop_size, "Training crop size")
    network.validate_crop_size(args.test_crop_size, "Test crop size")  # type: ignore
    # Initialize network weights
    if run_weight_initialization:
        network.apply(init_weights)  # type: ignore
    return network


def summary_for_segmentation_models(config: ModelConfigBase, model: DeviceAwareModule) -> None:
    """
    Generates a human readable summary of the present segmentation model, writes it to logging.info, and
    stores the ModelSummary object inside the argument `model`.

    :param config: The configuration for the model.
    :param model: The instantiated Pytorch model.
    """
    assert isinstance(model, BaseSegmentationModel)
    crop_size = config.crop_size
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size, crop_size)
    try:
        model.generate_model_summary(crop_size, log_summaries_to_files=config.log_summaries_to_files)
    except AttributeError as e:
        logging.warning(f"summary_for_segmentation_models failed with exception {e}")


def generate_and_print_model_summary(config: ModelConfigBase, model: DeviceAwareModule) -> None:
    """
    Writes a human readable summary of the present model to logging.info, and logs the number of trainable
    parameters to AzureML.
    :param config: The configuration for the model.
    :param model: The instantiated Pytorch model.
    """
    random_state = RandomStateSnapshot.snapshot_random_state()
    # There appears to be a bug in apex, where previous use (in training for example) causes problems
    # when another model is later built on the CPU (for example, before loading from a checkpoint)
    # https://github.com/NVIDIA/apex/issues/694
    # Hence, move the model to the GPU before doing model summary.
    if config.use_gpu:
        model = model.cuda()
    if isinstance(config, ScalarModelBase):
        # To generate the model summary, read the first item of the dataset. Then use the model's own
        # get_model_input function to convert the dataset item to input tensors, and feed them through the model.
        train_dataset = config.get_torch_dataset_for_inference(ModelExecutionMode.TRAIN)
        train_item_0 = next(iter(train_dataset.as_data_loader(shuffle=False, batch_size=1, num_dataload_workers=0)))
        target_indices = config.get_target_indices() if isinstance(config, SequenceModelBase) else []
        model_inputs = get_scalar_model_inputs_and_labels(model,
                                                          target_indices=target_indices,
                                                          sample=train_item_0)
        # The model inputs may already be converted to float16, assuming that we would do mixed precision.
        # However, the model is not yet converted to float16 when this function is called, hence convert back to float32
        summary = ModelSummary(model)
        summary.generate_summary(input_tensors=model_inputs.model_inputs,
                                 log_summaries_to_files=config.log_summaries_to_files)
    elif config.is_segmentation_model:
        summary_for_segmentation_models(config, model)
        summary = model.summarizer  # type: ignore
    else:
        raise ValueError("Don't know how to generate a summary for this type of model?")
    RUN_CONTEXT.log(LoggingColumns.NumTrainableParameters, summary.n_trainable_params)
    random_state.restore_random_state()
    # Move model back to CPU, to not mess with where Lightning expects things.
    model.cpu()


def create_model_with_temperature_scaling(config: ModelConfigBase) -> Any:
    """
    Create a model with temperature scaling by wrapping the result of config.create_model with ModelWithTemperature,
    if temperature scaling config has been provided, otherwise return the result of config.create_model
    """
    # wrap the model around a temperature scaling model if required
    model = config.create_model()
    if isinstance(config, SequenceModelBase) and config.temperature_scaling_config:
        model = ModelWithTemperature(model, config.temperature_scaling_config)
    return model


E = TypeVar('E', List[ClassificationItemSequence[ScalarItem]], ScalarItem)


@dataclass
class ScalarModelInputsAndLabels(Generic[E]):
    """
    Holds the results of calling get_scalar_model_inputs_and_labels: For a given sample returned by the data loader,
    create the model inputs, the labels, the list of subjects (data loader sample can be batched),
    and the reconstructed data item.
    """
    model_inputs: List[torch.Tensor]
    labels: torch.Tensor
    subject_ids: List[str]
    data_item: E

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

    def move_to_device(self, device: Union[str, torch.device]) -> None:
        """
        Moves the model_inputs and labels field of the present object to the given device. This is done in-place.
        :param device: The target device.
        """
        self.model_inputs = [t.to(device=device) for t in self.model_inputs]
        self.labels = self.labels.to(device=device)


def get_scalar_model_inputs_and_labels(model: torch.nn.Module,
                                       target_indices: List[int],
                                       sample: Dict[str, Any]) -> ScalarModelInputsAndLabels:
    """
    For a model that predicts scalars, gets the model input tensors from a sample returned by the data loader.
    :param model: The instantiated PyTorch model.
    :param target_indices: If this list is non-empty, assume that the model is a sequence model, and build the
    model inputs and labels for a model that predicts those specific positions in the sequence. If the list is empty,
    assume that the model is a normal (non-sequence) model.
    :param sample: A training sample, as returned by a PyTorch data loader (dictionary mapping from field name to value)
    :return: An instance of ScalarModelInputsAndLabels, containing the list of model input tensors,
    label tensor, subject IDs, and the data item reconstructed from the data loader output
    """
    if target_indices:
        sequence_model: DeviceAwareModule[List[ClassificationItemSequence], torch.Tensor] = model  # type: ignore
        sequences = ClassificationItemSequence.from_minibatch(sample)
        subject_ids = [x.id for x in sequences]
        labels = ClassificationItemSequence.create_labels_tensor_for_minibatch(
            sequences=sequences,
            target_indices=target_indices
        )
        model_inputs = sequence_model.get_input_tensors(sequences)

        return ScalarModelInputsAndLabels[List[ClassificationItemSequence]](
            model_inputs=model_inputs,
            labels=labels,
            subject_ids=subject_ids,
            data_item=sequences
        )
    else:
        scalar_model: DeviceAwareModule[ScalarItem, torch.Tensor] = model  # type: ignore
        scalar_item = ScalarItem.from_dict(sample)
        subject_ids = [str(x.id) for x in scalar_item.metadata]  # type: ignore
        model_inputs = scalar_model.get_input_tensors(scalar_item)

        return ScalarModelInputsAndLabels[ScalarItem](
            model_inputs=model_inputs,
            labels=scalar_item.label,
            subject_ids=subject_ids,
            data_item=scalar_item
        )

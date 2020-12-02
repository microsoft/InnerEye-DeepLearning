#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generic, Iterator, List, Optional, TypeVar, Union

import torch
from pytorch_lightning import LightningModule
from torch.nn import MSELoss
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.Common import common_util
from InnerEye.Common.type_annotations import T
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import ModelArchitectureConfig, PaddingMode, SegmentationLoss, SegmentationModelBase, \
    basic_size_shrinkage
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence
from InnerEye.ML.deep_learning_config import DeepLearningConfig, OptimizerType
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.architectures.complex import ComplexModel
from InnerEye.ML.models.architectures.unet_2d import UNet2D
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.losses.cross_entropy import CrossEntropyLoss
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.models.losses.soft_dice import SoftDiceLoss
from InnerEye.ML.models.parallel.data_parallel import DataParallelModel
from InnerEye.ML.scalar_config import ScalarLoss, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.metrics_constants import LoggingColumns
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
        return BinaryCrossEntropyWithLogitsLoss(smoothing_eps=config.label_smoothing_eps)
    if config.loss_type == ScalarLoss.WeightedCrossEntropyWithLogits:
        return BinaryCrossEntropyWithLogitsLoss(
            smoothing_eps=config.label_smoothing_eps,
            class_counts=config.get_training_class_counts())
    elif config.loss_type == ScalarLoss.MeanSquaredError:
        return MSELoss()
    else:
        raise NotImplementedError(f"Loss type {config.loss_type} is not implemented")


class ModelAndInfo:
    """
    This class contains the model and optional associated information, as well as methods to create
    models and optimizers, move these to GPU and load state from checkpoints. Attributes are:
      config: the model configuration information
      model: the model created based on the config
      optimizer: the optimizer created based on the config and associated with the model
      checkpoint_path: the path load load checkpoint from, can be None
      mean_teacher_model: the mean teacher model, if and as specified by the config
      is_model_adjusted: whether model adjustments (which cannot be done twice) have been applied to model
      is_mean_teacher_model_adjusted: whether model adjustments (which cannot be done twice)
      have been applied to the mean teacher model
      checkpoint_epoch: the training epoch this model was created, if loaded from disk
      model_execution_mode: mode this model will be run in
    """

    MODEL_STATE_DICT_KEY = 'state_dict'
    OPTIMIZER_STATE_DICT_KEY = 'opt_dict'
    MEAN_TEACHER_STATE_DICT_KEY = 'mean_teacher_state_dict'
    EPOCH_KEY = 'epoch'

    def __init__(self,
                 config: ModelConfigBase,
                 model_execution_mode: ModelExecutionMode,
                 checkpoint_path: Optional[Path] = None):
        """
        :param config: the model configuration information
        :param model_execution_mode: mode this model will be run in
        :param checkpoint_path: the path load load checkpoint from, can be None
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.model_execution_mode = model_execution_mode

        self._model = None
        self._mean_teacher_model = None
        self._optimizer = None
        self.checkpoint_epoch = None
        self.is_model_adjusted = False
        self.is_mean_teacher_model_adjusted = False

    @property
    def model(self) -> DeviceAwareModule:
        if not self._model:
            raise ValueError("Model has not been created.")
        return self._model

    @property
    def optimizer(self) -> Optimizer:
        if not self._optimizer:
            raise ValueError("Optimizer has not been created.")
        return self._optimizer

    @property
    def mean_teacher_model(self) -> Optional[DeviceAwareModule]:
        if not self._mean_teacher_model and self.config.compute_mean_teacher_model:
            raise ValueError("Mean teacher model has not been created.")
        return self._mean_teacher_model

    @staticmethod
    def read_checkpoint(path_to_checkpoint: Path, use_gpu: bool) -> Dict[str, Any]:
        # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
        # if the model is small.
        map_location = None if use_gpu else 'cpu'
        checkpoint = torch.load(str(path_to_checkpoint), map_location=map_location)
        return checkpoint

    @classmethod
    def _load_checkpoint(cls, model: LightningModule, checkpoint_path: Path,
                         key_in_state_dict: str, use_gpu: bool) -> int:
        """
        Loads a checkpoint of a model, may be the model or the mean teacher model. Assumes the model
        has already been created, and the checkpoint exists. This does not set checkpoint epoch.
        This method should not be called externally. Use instead try_load_checkpoint_for_model
        or try_load_checkpoint_for_mean_teacher_model
        :param model: model to load weights
        :param checkpoint_path: Path to checkpoint
        :param key_in_state_dict: the key for the model weights in the checkpoint state dict
        :param reader: Function which takes the path and returns a dict with model and optimizer states
        :return checkpoint epoch from the state dict
        """
        logging.info(f"Loading checkpoint {checkpoint_path}")
        checkpoint = ModelAndInfo.read_checkpoint(checkpoint_path, use_gpu)

        try:
            state_dict = checkpoint[key_in_state_dict]
        except KeyError:
            logging.error(f"Key {key_in_state_dict} not found in checkpoint")
            return False

        if isinstance(model, torch.nn.DataParallel):
            result = model.module.load_state_dict(state_dict, strict=False)
        else:
            result = model.load_state_dict(state_dict, strict=False)

        if result.missing_keys:
            logging.warning(f"Missing keys in model checkpoint: {result.missing_keys}")
        if result.unexpected_keys:
            logging.warning(f"Unexpected keys in model checkpoint: {result.unexpected_keys}")

        return checkpoint[ModelAndInfo.EPOCH_KEY]

    @classmethod
    def _adjust_for_gpus(cls, model: DeviceAwareModule, config: ModelConfigBase,
                         model_execution_mode: ModelExecutionMode) -> DeviceAwareModule:
        """
        Updates a torch model so that input mini-batches are parallelized across the batch dimension to utilise
        multiple gpus. If model parallel is set to True and execution is in test mode, then model is partitioned to
        perform full volume inference.
        This assumes the model has been created, that the optimizer has not yet been created, and the the model has not
        been adjusted twice. This method should not be called externally. Use instead adjust_model_for_gpus
        or adjust_mean_teacher_model_for_gpus
        :returns Adjusted model
        """
        if config.use_gpu:
            model = model.cuda()
            logging.info("Adjusting the model to use mixed precision training.")
            # If model parallel is set to True, then partition the network across all available gpus.
            if config.use_model_parallel:
                devices = config.get_cuda_devices()
                assert devices is not None  # for mypy
                model.partition_model(devices=devices)  # type: ignore
        else:
            logging.info("Making no adjustments to the model because no GPU was found.")

        # Update model related config attributes (After Model Parallel Activated)
        config.adjust_after_mixed_precision_and_parallel(model)

        # DataParallel enables running the model with multiple gpus by splitting samples across GPUs
        # If the model is used in training mode, data parallel is activated by default.
        # Similarly, if model parallel is not activated, data parallel is used as a backup option
        use_data_parallel = (model_execution_mode == ModelExecutionMode.TRAIN) or (not config.use_model_parallel)
        if config.use_gpu and use_data_parallel:
            logging.info("Adjusting the model to use DataParallel")
            # Move all layers to the default GPU before activating data parallel.
            # This needs to happen even though we put the model to the GPU at the beginning of the method,
            # but we may have spread it across multiple GPUs later.
            model = model.cuda()
            model = DataParallelModel(model, device_ids=config.get_cuda_devices())

        return model

    def create_model(self) -> None:
        """
        Creates a model (with temperature scaling) according to the config given.
        """
        # self._model = create_model_with_temperature_scaling(self.config)
        # self._model = create_lightning_model(self.config)

    def try_load_checkpoint_for_model(self) -> bool:
        """
        Loads a checkpoint of a model. The provided model checkpoint must match the stored model.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        if self._model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        if not self.checkpoint_path:
            raise ValueError("No checkpoint provided")

        if not self.checkpoint_path.is_file():
            logging.warning(f'No checkpoint found at {self.checkpoint_path} current working dir {os.getcwd()}')
            return False

        epoch = ModelAndInfo._load_checkpoint(model=self._model,
                                              checkpoint_path=self.checkpoint_path,
                                              key_in_state_dict=ModelAndInfo.MODEL_STATE_DICT_KEY,
                                              use_gpu=self.config.use_gpu)

        logging.info(f"Loaded model from checkpoint (epoch: {epoch})")
        self.checkpoint_epoch = epoch
        return True

    def adjust_model_for_gpus(self) -> None:
        """
        Updates the torch model so that input mini-batches are parallelized across the batch dimension to utilise
        multiple gpus. If model parallel is set to True and execution is in test mode, then model is partitioned to
        perform full volume inference.
        """
        if self._model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        # Adjusting twice causes an error.
        if self.is_model_adjusted:
            logging.debug("model_and_info.is_model_adjusted is already True")

        if self._optimizer:
            raise ValueError("Create an optimizer only after creating and adjusting the model.")

        self._model = ModelAndInfo._adjust_for_gpus(model=self._model,
                                                    config=self.config,
                                                    model_execution_mode=self.model_execution_mode)

        self.is_model_adjusted = True
        logging.debug("model_and_info.is_model_adjusted set to True")

    def create_summary_and_adjust_model_for_gpus(self) -> None:
        """
        Generates the model summary, which is required for model partitioning across GPUs, and then moves the model to
        GPU with data parallel/model parallel by calling adjust_model_for_gpus.
        """
        if self._model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        if self.config.is_segmentation_model:
            summary_for_segmentation_models(self.config, self._model)
        # Prepare for mixed precision training and data parallelization (no-op if already done).
        # This relies on the information generated in the model summary.
        self.adjust_model_for_gpus()

    def try_create_model_and_load_from_checkpoint(self) -> bool:
        """
        Creates a model as per the config, and loads the parameters from the given checkpoint path.
        Also updates the checkpoint_epoch.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        self.create_model()
        if self.checkpoint_path:
            # Load the stored model. If there is no checkpoint present, return immediately.
            return self.try_load_checkpoint_for_model()
        return True

    def try_create_model_load_from_checkpoint_and_adjust(self) -> bool:
        """
        Creates a model as per the config, and loads the parameters from the given checkpoint path.
        The model is then adjusted for data parallelism and mixed precision.
        Also updates the checkpoint_epoch.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        success = self.try_create_model_and_load_from_checkpoint()
        # TODO antonsc: undo hack
        # self.create_summary_and_adjust_model_for_gpus()
        self.config.adjust_after_mixed_precision_and_parallel(self.model.model)
        return success

    def create_mean_teacher_model(self) -> None:
        """
        Creates a model (with temperature scaling) according to the config given.
        """
        self._mean_teacher_model = create_model_with_temperature_scaling(self.config)

    def try_load_checkpoint_for_mean_teacher_model(self) -> bool:
        """
        Loads a checkpoint of a model. The provided model checkpoint must match the stored model.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        if self._mean_teacher_model is None:
            raise ValueError("Mean teacher model must be created before it can be adjusted.")

        if not self.checkpoint_path:
            raise ValueError("No checkpoint provided")

        if not self.checkpoint_path.is_file():
            logging.warning(f'No checkpoint found at {self.checkpoint_path} current working dir {os.getcwd()}')
            return False

        epoch = ModelAndInfo._load_checkpoint(model=self._mean_teacher_model,
                                              checkpoint_path=self.checkpoint_path,
                                              key_in_state_dict=ModelAndInfo.MEAN_TEACHER_STATE_DICT_KEY,
                                              use_gpu=self.config.use_gpu)

        logging.info(f"Loaded mean teacher model from checkpoint (epoch: {epoch})")
        self.checkpoint_epoch = epoch
        return True

    def adjust_mean_teacher_model_for_gpus(self) -> None:
        """
        Updates the torch model so that input mini-batches are parallelized across the batch dimension to utilise
        multiple gpus. If model parallel is set to True and execution is in test mode, then model is partitioned to
        perform full volume inference.
        """
        if self._mean_teacher_model is None:
            raise ValueError("Mean teacher model must be created before it can be adjusted.")

        # Adjusting twice causes an error.
        if self.is_mean_teacher_model_adjusted:
            logging.debug("model_and_info.is_mean_teacher_model_adjusted is already True")

        self._mean_teacher_model = ModelAndInfo._adjust_for_gpus(model=self._mean_teacher_model,
                                                                 config=self.config,
                                                                 model_execution_mode=self.model_execution_mode)

        self.is_mean_teacher_model_adjusted = True
        logging.debug("model_and_info.is_mean_teacher_model_adjusted set to True")

    def create_summary_and_adjust_mean_teacher_model_for_gpus(self) -> None:
        """
        Generates the model summary, which is required for model partitioning across GPUs, and then moves the model to
        GPU with data parallel/model parallel by calling adjust_model_for_gpus.
        """
        if self._mean_teacher_model is None:
            raise ValueError("Mean teacher model must be created before it can be adjusted.")

        if self.config.is_segmentation_model:
            summary_for_segmentation_models(self.config, self._mean_teacher_model)
        # Prepare for mixed precision training and data parallelization (no-op if already done).
        # This relies on the information generated in the model summary.
        self.adjust_mean_teacher_model_for_gpus()

    def try_create_mean_teacher_model_and_load_from_checkpoint(self) -> bool:
        """
        Creates a model as per the config, and loads the parameters from the given checkpoint path.
        Also updates the checkpoint_epoch.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        self.create_mean_teacher_model()
        if self.checkpoint_path:
            # Load the stored model. If there is no checkpoint present, return immediately.
            return self.try_load_checkpoint_for_mean_teacher_model()
        return True

    def try_create_mean_teacher_model_load_from_checkpoint_and_adjust(self) -> bool:
        """
        Creates a model as per the config, and loads the parameters from the given checkpoint path.
        The model is then adjusted for data parallelism and mixed precision.
        Also updates the checkpoint_epoch.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        success = self.try_create_mean_teacher_model_and_load_from_checkpoint()
        self.create_summary_and_adjust_mean_teacher_model_for_gpus()
        return success

    def create_optimizer(self) -> None:
        """
        Creates a torch optimizer for the given model, and stores it as an instance variable in the current object.
        """
        # Make sure model is created before we create optimizer
        if self._model is None:
            raise ValueError("Model checkpoint must be created before optimizer checkpoint can be loaded.")
        self._optimizer = create_optimizer(self.config, self._model.parameters())

    def try_load_checkpoint_for_optimizer(self) -> bool:
        """
        Loads a checkpoint of an optimizer.
        :return True if the checkpoint exists and optimizer state loaded, False otherwise
        """

        if self._optimizer is None:
            raise ValueError("Optimizer must be created before optimizer checkpoint can be loaded.")

        if not self.checkpoint_path:
            logging.warning("No checkpoint path provided.")
            return False

        if not self.checkpoint_path.is_file():
            logging.warning(f'No checkpoint found at {self.checkpoint_path} current working dir {os.getcwd()}')
            return False

        logging.info(f"Loading checkpoint {self.checkpoint_path}")
        checkpoint = ModelAndInfo.read_checkpoint(self.checkpoint_path, self.config.use_gpu)

        try:
            state_dict = checkpoint[ModelAndInfo.OPTIMIZER_STATE_DICT_KEY]
        except KeyError:
            logging.error(f"Key {ModelAndInfo.OPTIMIZER_STATE_DICT_KEY} not found in checkpoint")
            return False

        self._optimizer.load_state_dict(state_dict)

        logging.info(f"Loaded optimizer from checkpoint (epoch: {checkpoint[ModelAndInfo.EPOCH_KEY]})")
        self.checkpoint_epoch = checkpoint[ModelAndInfo.EPOCH_KEY]
        return True

    def try_create_optimizer_and_load_from_checkpoint(self) -> bool:
        """
        Creates an optimizer and loads its state from a checkpoint.
        :return True if the checkpoint exists and optimizer state loaded, False otherwise
        """
        self.create_optimizer()
        if self.checkpoint_path:
            return self.try_load_checkpoint_for_optimizer()
        return True

    def save_checkpoint(self, epoch: int) -> Path:
        """
        Saves a checkpoint of the current model and optimizer_type parameters in the specified folder
        and uploads it to the output blob storage of the current run context.
        The checkpoint's name for epoch 123 would be 123_checkpoint.pth.tar.
        :param epoch: The last epoch used to train the model.
        :return: The full path of the checkpoint file.
        """
        logging.getLogger().disabled = True
        model_state_dict = self.model.module.state_dict() \
            if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        checkpoint_file_path = self.config.get_path_to_checkpoint(epoch)
        checkpoint_file_path.parent.mkdir(exist_ok=True, parents=True)
        info_to_store = {
            ModelAndInfo.EPOCH_KEY: epoch,
            ModelAndInfo.MODEL_STATE_DICT_KEY: model_state_dict,
            ModelAndInfo.OPTIMIZER_STATE_DICT_KEY: self.optimizer.state_dict()
        }
        if self.config.compute_mean_teacher_model:
            assert self.mean_teacher_model is not None  # for mypy, getter has this built in
            mean_teacher_model_state_dict = self.mean_teacher_model.module.state_dict() \
                if isinstance(self.mean_teacher_model, torch.nn.DataParallel) \
                else self.mean_teacher_model.state_dict()
            info_to_store[ModelAndInfo.MEAN_TEACHER_STATE_DICT_KEY] = mean_teacher_model_state_dict

        torch.save(info_to_store, checkpoint_file_path)
        logging.getLogger().disabled = False
        logging.info(f"Saved model checkpoint for epoch {epoch} to {checkpoint_file_path}")
        return checkpoint_file_path


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
def build_net(args: SegmentationModelBase) -> BaseModel:
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

    network: BaseModel
    if args.architecture == ModelArchitectureConfig.Basic:
        network_definition = basic_network_definition
        network = ComplexModel(args, full_channels_list,
                               basic_dilations, network_definition, crop_size_constraints)  # type: ignore

    elif args.architecture == ModelArchitectureConfig.UNet3D:
        network = UNet3D(input_image_channels=args.number_of_image_channels,
                         initial_feature_channels=args.feature_channels[0],
                         num_classes=args.number_of_classes,
                         kernel_size=args.kernel_size)
        run_weight_initialization = False

    elif args.architecture == ModelArchitectureConfig.UNet2D:
        network = UNet2D(input_image_channels=args.number_of_image_channels,
                         initial_feature_channels=args.feature_channels[0],
                         num_classes=args.number_of_classes,
                         padding_mode=PaddingMode.Edge)
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
    assert isinstance(model, BaseModel)
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
        model_inputs = get_scalar_model_inputs_and_labels(config, model, train_item_0).model_inputs
        # The model inputs may already be converted to float16, assuming that we would do mixed precision.
        # However, the model is not yet converted to float16 when this function is called, hence convert back to float32
        summary = ModelSummary(model)
        summary.generate_summary(input_tensors=model_inputs, log_summaries_to_files=config.log_summaries_to_files)
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
    if isinstance(model_config, SequenceModelBase):
        sequence_model: DeviceAwareModule[List[ClassificationItemSequence], torch.Tensor] = model  # type: ignore
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
        scalar_model: DeviceAwareModule[ScalarItem, torch.Tensor] = model  # type: ignore
        scalar_item = ScalarItem.from_dict(sample)
        subject_ids = [str(x.id) for x in scalar_item.metadata]  # type: ignore
        model_inputs = scalar_model.get_input_tensors(scalar_item)

        return ScalarModelInputsAndLabels[ScalarItem, torch.Tensor](
            model_inputs=model_inputs,
            labels=scalar_item.label,
            subject_ids=subject_ids,
            data_item=scalar_item
        )

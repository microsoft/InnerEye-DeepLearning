#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.rmsprop import RMSprop

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import ModelArchitectureConfig, PaddingMode, SegmentationModelBase, \
    basic_size_shrinkage
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training_steps import get_scalar_model_inputs_and_labels
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.architectures.complex import ComplexModel
from InnerEye.ML.models.architectures.unet_2d import UNet2D
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.models.layers.basic import BasicLayer
from InnerEye.ML.models.parallel.data_parallel import DataParallelModel
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, is_gpu_available
from InnerEye.ML.utils.temperature_scaling import ModelWithTemperature
from InnerEye.ML.visualizers.model_summary import ModelSummary

BaseModelOrDataParallelModel = Union[DeviceAwareModule, DataParallelModel]


class ModelAndInfo:
    """
    A holder for a model and, optionally, associated information.
      model: any model
      optimizer: associated optimizer if any
      is_mean_teacher: whether this is (intended to be) a mean teacher model
      is_adjusted: whether model adjustments (which cannot be done twice) have been applied
      checkpoint_epoch: the training epoch this model was created, if loaded from disk
      model_execution_mode: mode this model will be run in
    """
    def __init__(self,
                 config: ModelConfigBase,
                 model_execution_mode: ModelExecutionMode,
                 is_mean_teacher: bool = False,
                 checkpoint_path: Path = None):
        self.config = config
        self.is_mean_teacher = is_mean_teacher
        self.checkpoint_path = checkpoint_path
        self.model_execution_mode = model_execution_mode

        self.model = None
        self.optimizer = None
        self.checkpoint_epoch = None
        self.is_adjusted = False

    def to_cuda(self) -> None:
        """
        Moves the model to GPU
        """
        if self.model is None:
            raise ValueError("Model must be created before it can be moved to GPU.")
        self.model = self.model.cuda()

    def set_data_parallel(self, device_ids: Optional[List[Any]]) -> None:
        if self.model is None:
            raise ValueError("Model must be created before it can be moved to Data Parellel.")
        self.model = DataParallelModel(self.model, device_ids=device_ids)

    def create_model(self) -> None:
        """
        Creates a model (with temperature scaling) according to the config given.
        """
        self.model = create_model_with_temperature_scaling(self.config)

    def try_load_checkpoint_for_model(self) -> bool:
        """
        Loads a checkpoint of a model. The provided model checkpoint must match the stored model.
        """
        if self.model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        if not self.checkpoint_path:
            raise ValueError("No checkpoint provided")

        if not self.checkpoint_path.is_file():
            logging.warning(f'No checkpoint found at {self.checkpoint_path} current working dir {os.getcwd()}')
            return False

        logging.info(f"Loading checkpoint {self.checkpoint_path}")
        # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
        # if the model is small.
        map_location = None if is_gpu_available() else 'cpu'
        checkpoint = torch.load(str(self.checkpoint_path), map_location=map_location)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        logging.info("Loaded model from checkpoint (epoch: {})".format(checkpoint['epoch']))
        self.checkpoint_epoch = checkpoint['epoch']
        return True

    def adjust_model_for_gpus(self) -> None:
        """
        Updates the torch model so that input mini-batches are parallelized across the batch dimension to utilise
        multiple gpus. If model parallel is set to True and execution is in test mode, then model is partitioned to
        perform full volume inference.
        """
        if self.model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        # Adjusting twice causes an error.
        if self.is_adjusted:
            logging.debug("model_and_info.is_adjusted is already True")

        if self.optimizer is not None:
            raise ValueError("Create an optimizer only after creating and adjusting the model.")

        if self.config.use_gpu:
            self.to_cuda()
            logging.info("Adjusting the model to use mixed precision training.")
            # If model parallel is set to True, then partition the network across all available gpus.
            if self.config.use_model_parallel:
                devices = self.config.get_cuda_devices()
                assert devices is not None  # for mypy
                self.model.partition_model(devices=devices)  # type: ignore
        else:
            logging.info("Making no adjustments to the model because no GPU was found.")

        # Update model related config attributes (After Model Parallel Activated)
        self.config.adjust_after_mixed_precision_and_parallel(self.model)

        # DataParallel enables running the model with multiple gpus by splitting samples across GPUs
        # If the model is used in training mode, data parallel is activated by default.
        # Similarly, if model parallel is not activated, data parallel is used as a backup option
        use_data_parallel = (self.model_execution_mode == ModelExecutionMode.TRAIN) or (not self.config.use_model_parallel)
        if self.config.use_gpu and use_data_parallel:
            logging.info("Adjusting the model to use DataParallel")
            # Move all layers to the default GPU before activating data parallel.
            # This needs to happen even though we put the model to the GPU at the beginning of the method,
            # but we may have spread it across multiple GPUs later.
            self.to_cuda()
            self.set_data_parallel(device_ids=self.config.get_cuda_devices())

        self.is_adjusted = True
        logging.debug("model_and_info.is_adjusted set to True")

    def create_summary_and_adjust_model_for_gpus(self) -> None:
        """
        Generates the model summary, which is required for model partitioning across GPUs, and then moves the model to
        GPU with data parallel/model parallel by calling adjust_model_for_gpus.
        """
        if self.model is None:
            raise ValueError("Model must be created before it can be adjusted.")

        if self.config.is_segmentation_model:
            summary_for_segmentation_models(self.config, self.model)
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

        # for mypy
        assert self.model is not None

        if self.checkpoint_path is not None:
            # Load the stored model. If there is no checkpoint present, return immediately.
            success = self.try_load_checkpoint_for_model()
            if not success:
                return False
        return True

    def try_create_model_load_from_checkpoint_and_adjust(self) -> bool:
        """
        Creates a model as per the config, and loads the parameters from the given checkpoint path.
        The model is then adjusted for data parallelism and mixed precision, running in TEST mode.
        Also updates the checkpoint_epoch.
        :return True if checkpoint exists and was loaded, False otherwise.
        """
        self.create_model()

        # for mypy
        assert self.model is not None

        if self.checkpoint_path is not None:
            # Load the stored model. If there is no checkpoint present, return immediately.
            success = self.try_load_checkpoint_for_model()
            if not success:
                return False
        self.create_summary_and_adjust_model_for_gpus()
        return True

    def create_optimizer(self) -> None:
        """
        Creates a torch optimizer for the given model.
        """
        # Make sure model is created before we create optimizer
        if self.model is None:
            raise ValueError("Model checkpoint must be created before optimizer checkpoint can be loaded.")

        # Select optimizer type
        if self.config.optimizer_type in [OptimizerType.Adam, OptimizerType.AMSGrad]:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.config.l_rate,
                                              self.config.adam_betas, self.config.opt_eps, self.config.weight_decay,
                                              amsgrad=self.config.optimizer_type == OptimizerType.AMSGrad)
        elif self.config.optimizer_type == OptimizerType.SGD:
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.config.l_rate, self.config.momentum,
                                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer_type == OptimizerType.RMSprop:
            self.optimizer = RMSprop(self.model.parameters(), self.config.l_rate, self.config.rms_alpha, self.config.opt_eps,
                           self.config.weight_decay, self.config.momentum)
        else:
            raise NotImplementedError(f"Optimizer type {self.config.optimizer_type.value} is not implemented")

    def try_load_checkpoint_for_optimizer(self) -> bool:
        """
        Loads a checkpoint of an optimizer.
        :return True if the checkpoint exists and optimizer state loaded, False otherwise
        """

        if self.optimizer is None:
            raise ValueError("Optimizer must be created before optimizer checkpoint can be loaded.")

        if not self.checkpoint_path.is_file():
            logging.warning(f'No checkpoint found at {self.checkpoint_path} current working dir {os.getcwd()}')
            return False

        logging.info(f"Loading checkpoint {self.checkpoint_path}")
        # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
        # if the model is small.
        map_location = None if is_gpu_available() else 'cpu'
        checkpoint = torch.load(str(self.checkpoint_path), map_location=map_location)

        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['opt_dict'])

        logging.info("Loaded optimizer from checkpoint (epoch: {})".format(checkpoint['epoch']))
        self.checkpoint_epoch = checkpoint['epoch']
        return True

    def try_create_optimizer_and_load_from_checkpoint(self) -> bool:
        """
        Creates an optimizer and loads its state from a checkpoint.
        :return True if the checkpoint exists and optimizer state loaded, False otherwise
        """
        self.create_optimizer()
        if self.checkpoint_path is not None:
            success = self.try_load_checkpoint_for_optimizer()
            if not success:
                return False
        return True


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
        raise ValueError("Unknown model architecture {}".format(args.architecture))
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
        assert model.summarizer
        summary = model.summarizer  # type: ignore
    else:
        raise ValueError("Don't know how to generate a summary for this type of model?")
    RUN_CONTEXT.log(LoggingColumns.NumTrainableParameters, summary.n_trainable_params)
    random_state.restore_random_state()


def save_checkpoint(model: torch.nn.Module, optimizer: Optimizer, epoch: int,
                    args: ModelConfigBase, mean_teacher_model: bool = False) -> None:
    """
    Saves a checkpoint of the current model and optimizer_type parameters in the specified folder
    and uploads it to the output blob storage of the current run context.
    The checkpoint's name for epoch 123 would be 123_checkpoint.pth.tar.

    :param model: A DataParallel object representing the model.
    :param optimizer: The optimizer_type used for training.
    :param epoch: The last epoch used to train the model.
    :param args:
    :param mean_teacher_model: If True save to the mean teacher model checkpoint path.
    """
    logging.getLogger().disabled = True

    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    checkpoint_file_path = args.get_path_to_checkpoint(epoch, mean_teacher_model)
    info_to_store = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'opt_dict': optimizer.state_dict()
    }
    torch.save(info_to_store, checkpoint_file_path)
    logging.getLogger().disabled = False
    logging.info("Saved model checkpoint for epoch {} to {}".format(epoch, checkpoint_file_path))


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

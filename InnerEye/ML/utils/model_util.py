#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from apex import amp
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
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, is_gpu_available
from InnerEye.ML.visualizers.model_summary import ModelSummary

BaseModelOrDataParallelModel = Union[BaseModel, DataParallelModel]


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
def build_net(args: SegmentationModelBase) -> BaseModel:  # type: ignore
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

    if args.architecture == ModelArchitectureConfig.Basic:
        network_definition = basic_network_definition
        network = ComplexModel(args, full_channels_list,
                               basic_dilations, network_definition, crop_size_constraints)  # type: ignore

    elif args.architecture == ModelArchitectureConfig.UNet3D:
        network = UNet3D(input_image_channels=args.number_of_image_channels,  # type: ignore
                         initial_feature_channels=args.feature_channels[0],
                         num_classes=args.number_of_classes,
                         kernel_size=args.kernel_size)
        run_weight_initialization = False

    elif args.architecture == ModelArchitectureConfig.UNet2D:
        network = UNet2D(input_image_channels=args.number_of_image_channels,  # type: ignore
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


def update_model_for_mixed_precision_and_parallel(model: BaseModel,
                                                  args: ModelConfigBase,
                                                  optimizer: Optional[Optimizer] = None,
                                                  execution_mode: ModelExecutionMode = ModelExecutionMode.TRAIN) -> \
        Tuple[BaseModelOrDataParallelModel, Optional[Optimizer]]:
    """
    Updates a given torch model as such input mini-batches are parallelized across the batch dimension to utilise
    multiple gpus. If model parallel is set to True and execution is in test mode, then model is partitioned to
    perform full volume inference. Additionally, mixed precision training (amp) is utilised on both the model and
    optimizer instances to improve the training performance.

    :param model: The torch module object representing the network.
    :param args: The arguments object with attributes used to enable amp training and create the parallel model.
    :param optimizer: The torch optimizer that should be used for training.
    :return: Updated torch model and optimizer.
    """
    if args.use_gpu:
        # In the normal training codepath, the model should already be on the GPU, but in some tests not.
        model = model.cuda()
        logging.info("Adjusting the model to use mixed precision training.")
        # If model parallel is set to True, then partition the network across all available gpus.
        if args.use_model_parallel:
            devices = args.get_cuda_devices()
            assert devices is not None  # for mypy
            model.partition_model(devices=devices)

        # This is required to support sigmoid function
        amp.register_float_function(torch, 'sigmoid')

        # Activate automatic mixed precision
        # With optimization GEMMs and convolutions are performed in FP16, see https://nvidia.github.io/apex/amp.html
        amp_output = amp.initialize(model, optimizer, enabled=args.use_mixed_precision, opt_level="O1",
                                    keep_batchnorm_fp32=None, loss_scale="dynamic", num_losses=1)

        if isinstance(amp_output, tuple):
            model, optimizer = amp_output
        else:
            model = amp_output
    else:
        logging.info("Making no adjustments to the model because no GPU was found.")

    # Update model related config attributes (After AMP & Model Parallel Activated)
    args.adjust_after_mixed_precision_and_parallel(model)

    # DataParallel enables running the model with multiple gpus by splitting samples across GPUs
    # If the model is used in training mode, data parallel is activated by default.
    # Similarly, if model parallel is not activated, data parallel is used as a backup option
    use_data_parallel = (execution_mode == ModelExecutionMode.TRAIN) or (not args.use_model_parallel)
    if args.use_gpu and use_data_parallel:
        logging.info("Adjusting the model to use DataParallel")
        # Move all layers to the default GPU before activating data parallel.
        # This needs to happen even though we put the model to the GPU at the beginning of the method,
        # but we may have spread it across multiple GPUs later.
        model = model.cuda()
        model = DataParallelModel(model, device_ids=args.get_cuda_devices())  # type: ignore

    return model, optimizer


def create_optimizer(args: ModelConfigBase, model: torch.nn.Module) -> Optimizer:
    """
    Creates a torch optimizer for the given model.

    :param args: The arguments object with attributes used to create the optimizer_type.
    :param model: The DataParallel object representing the network.
    :return: An instance of torch.optim.Optimizer
    """
    # Select optimizer type
    if args.optimizer_type in [OptimizerType.Adam, OptimizerType.AMSGrad]:
        return torch.optim.Adam(model.parameters(), args.l_rate, args.adam_betas, args.opt_eps, args.weight_decay,
                                amsgrad=args.optimizer_type == OptimizerType.AMSGrad)
    elif args.optimizer_type == OptimizerType.SGD:
        return torch.optim.SGD(model.parameters(), args.l_rate, args.momentum,
                               weight_decay=args.weight_decay)
    elif args.optimizer_type == OptimizerType.RMSprop:
        return RMSprop(model.parameters(), args.l_rate, args.rms_alpha, args.opt_eps,
                                   args.weight_decay, args.momentum)
    else:
        raise NotImplementedError(f"Optimizer type {args.optimizer_type.value} is not implemented")


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
        model.generate_model_summary(crop_size)  # type: ignore
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
    if config.is_scalar_model:
        assert isinstance(config, ScalarModelBase)
        # To generate the model summary, read the first item of the dataset. Then use the model's own
        # get_model_input function to convert the dataset item to input tensors, and feed them through the model.
        train_dataset = config.get_torch_dataset_for_inference(ModelExecutionMode.TRAIN)
        train_item_0 = next(iter(train_dataset.as_data_loader(shuffle=False, batch_size=1, num_dataload_workers=0)))
        model_inputs = get_scalar_model_inputs_and_labels(config, model, train_item_0).model_inputs
        # The model inputs may already be converted to float16, assuming that we would do mixed precision.
        # However, the model is not yet converted to float16 when this function is called, hence convert back to float32
        if config.use_gpu:
            model_inputs = [x.float() for x in model_inputs]
        summary = ModelSummary(model)
        summary.generate_summary(input_tensors=model_inputs, log_models_to_files=config.log_models_to_files)
    elif config.is_segmentation_model:
        summary_for_segmentation_models(config, model)
        assert model.summarizer
        summary = model.summarizer  # type: ignore
    else:
        raise ValueError("Don't know how to generate a summary for this type of model?")
    RUN_CONTEXT.log(LoggingColumns.NumTrainableParameters, summary.n_trainable_params)
    random_state.restore_random_state()


def load_checkpoint(model: torch.nn.DataParallel,
                    path_to_checkpoint: Path,
                    optimizer: Optimizer = None) -> Optional[int]:
    """
    Loads a checkpoint of a model.
    The epoch of the stored model and the epoch provided as argument must match.
    The provided model must match the stored model.

    :param model: The DataParallel object representing the network. Must have the same architecture of the stored model.
    :param path_to_checkpoint: The path to the checkpoint file.
    :param optimizer: The optimizer used for training
    :return: The checkpoint epoch if loaded and None if not loaded
    """

    if not path_to_checkpoint.is_file():
        logging.warning(f'No checkpoint found at {path_to_checkpoint} current working dir {os.getcwd()}')
        return None

    logging.info(f"Loading checkpoint {path_to_checkpoint}")
    # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
    # if the model is small.
    map_location = None if is_gpu_available() else 'cpu'
    checkpoint = torch.load(str(path_to_checkpoint), map_location=map_location)

    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['opt_dict'])

    logging.info("Loaded checkpoint (epoch: {})".format(checkpoint['epoch']))
    return checkpoint['epoch']


def save_checkpoint(model: torch.nn.DataParallel, optimizer: Optimizer, epoch: int,
                    args: ModelConfigBase) -> None:
    """
    Saves a checkpoint of the current model and optimizer_type parameters in the specified folder
    and uploads it to the output blob storage of the current run context.
    The checkpoint's name for epoch 123 would be 123_checkpoint.pth.tar.

    :param model: A DataParallel object representing the model.
    :param optimizer: The optimizer_type used for training.
    :param epoch: The last epoch used to train the model.
    :param args:
    """
    logging.getLogger().disabled = True

    model_state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    checkpoint_file_path = args.get_path_to_checkpoint(epoch)
    info_to_store = {
        'epoch': epoch,
        'state_dict': model_state_dict,
        'opt_dict': optimizer.state_dict()
    }
    torch.save(info_to_store, checkpoint_file_path)
    logging.getLogger().disabled = False
    logging.info("Saved model checkpoint for epoch {} to {}".format(epoch, checkpoint_file_path))


def load_from_checkpoint_and_adjust(model_config: ModelConfigBase,
                                    path_to_checkpoint: Path) -> Optional[Tuple[BaseModelOrDataParallelModel, int]]:
    """
    Creates a model as per the configuration, and loads the parameters from the given checkpoint path.
    The model is then adjusted for data parallelism and mixed precision, running in TEST mode.

    :param model_config: The configuration from which an empty model will be created.
    :param path_to_checkpoint: The path to the checkpoint file.
    :return: The model with all loaded parameters, and the epoch in which the model was saved. If the result is None,
    there is no model file at the given path.
    """
    # Create model
    model = model_config.create_model()
    # Load the stored model. If there is not checkpoint present, return immediately.
    checkpoint_epoch = load_checkpoint(model=model,
                                       path_to_checkpoint=path_to_checkpoint,
                                       optimizer=None)
    if checkpoint_epoch is None:
        return None
    # Enable data/model parallelization
    if model_config.is_segmentation_model:
        # Generate the model summary, which is required for model partitioning across GPUs.
        summary_for_segmentation_models(model_config, model)
    model, _ = update_model_for_mixed_precision_and_parallel(model,
                                                             args=model_config,
                                                             optimizer=None,
                                                             execution_mode=ModelExecutionMode.TEST)
    return model, checkpoint_epoch

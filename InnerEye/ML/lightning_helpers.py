#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path

import torch

from InnerEye.ML.lightning_base import InnerEyeLightning
from InnerEye.ML.lightning_models import create_lightning_model
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel


def load_from_lightning_checkpoint(config: ModelConfigBase, checkpoint_path: Path) -> InnerEyeLightning:
    """
    Reads a PyTorch model from a checkpoint. First, a PyTorch Lightning model is created matching the InnerEye
    model configuration, its parameter tensors are then populated from the given checkpoint.
    :param config: An InnerEye model configuration object
    :param checkpoint_path: The location of the checkpoint file.
    :return: A PyTorch Lightning model object.
    """
    # Create a Lighting model that matches the configuration, but keep only the type of it
    lightning_model_type = type(create_lightning_model(config))
    # For model debugging, allow loading a GPU trained model onto the CPU. This will clearly only work
    # if the model is small.
    map_location = None if config.use_gpu else 'cpu'
    lightning_model = lightning_model_type.load_from_checkpoint(checkpoint_path=str(checkpoint_path),
                                                                map_location=map_location,
                                                                config=config)
    return lightning_model


def adjust_model_for_inference(config: ModelConfigBase, lightning_model: InnerEyeLightning) -> None:
    """
    Makes all necessary adjustments to use a given model for inference, possibly on multiple GPUs via
    model parallelization. The method also computes parameters like output patch size for segmentation model,
    and stores them in the model configuration.
    :param config: The model configuration object. It may be modified in place.
    :param lightning_model: The trained model that should be adjusted.
    """
    if config.use_gpu:
        lightning_model: InnerEyeLightning = lightning_model.cuda()  # type: ignore
        # If model parallel is set to True, then partition the network across all available gpus.
        # Model partitioning relies on the model summary. We generate that with a smaller crop (the same that is also
        # used during training, and we assume that fits onto the GPU)
        if config.use_model_parallel and isinstance(lightning_model.model, BaseSegmentationModel):
            logging.info("Partitioning the model across all GPUs.")
            lightning_model.model.generate_model_summary(crop_size=config.crop_size, log_summaries_to_files=True)
            lightning_model.model.partition_model()
    else:
        logging.info("Skipping model partitioning because no GPU was found.")

    # Update model related config attributes. This must happen after model partitioning, because we compute the
    # model output size during inference: That will only fit onto the GPU if already partitioned.
    used_gpus = set(p.device for p in lightning_model.parameters())
    logging.info(f"Model is using these devices: {used_gpus}")
    logging.info("Re-computing model-dependent properties (e.g., output patch sizes)")
    config.set_derived_model_properties(lightning_model.model)
    torch.cuda.empty_cache()


def load_from_checkpoint_and_adjust_for_inference(config: ModelConfigBase, checkpoint_path: Path) -> InnerEyeLightning:
    """
    Reads a PyTorch model from a checkpoint, and makes all necessary adjustments to use the model for inference,
    possibly on multiple GPUs.
    :param config: An InnerEye model configuration object
    :param checkpoint_path: The location of the checkpoint file.
    :return: A PyTorch Lightning model object.
    """
    lightning_model = load_from_lightning_checkpoint(config, checkpoint_path)
    lightning_model.eval()
    adjust_model_for_inference(config, lightning_model)
    return lightning_model

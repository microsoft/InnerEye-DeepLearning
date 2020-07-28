#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.visualizers import model_hooks


def vis_activation_map(activation_map: np.ndarray) -> np.ndarray:
    """
    Normalizes the activation map and maps it to RGB range for visualization
    :param activation_map:
    :return:
    """

    min_val = np.min(activation_map)
    activation_map += abs(min_val)

    # scale to RGB
    activation_map = (activation_map / np.max(activation_map)) * 255.0

    return activation_map


def visualize_2d_activation_map(activation_map: np.ndarray, args: ModelConfigBase, slice_index: int = 0) -> None:
    """
    Saves all feature channels of a 2D activation map as png files
    :param activation_map:
    :param args:
    :param slice_index:
    :return:
    """
    destination_directory = str(args.outputs_folder / "activation_maps")

    if not os.path.exists(destination_directory):
        os.mkdir(destination_directory)

    for feat in range(activation_map.shape[0]):
        plt.imshow(vis_activation_map(activation_map[feat]))
        plt.savefig(os.path.join(destination_directory,
                                 "slice_" + str(slice_index) + "_feature_" + (str(feat) + "_Activation_Map.png")))


def visualize_3d_activation_map(activation_map: np.ndarray, args: ModelConfigBase,
                                slices_to_visualize: Optional[List[int]] = None) -> None:
    """
    Saves all feature channels of a 3D activation map as png files
    :param activation_map:
    :param args:
    :param slices_to_visualize:
    :return:
    """

    # Only visualize some slices, random choice if not set
    if slices_to_visualize is None:
        slices_to_visualize = np.random.randint(0, activation_map.shape[1], 2).tolist()

    for _slice in slices_to_visualize:
        visualize_2d_activation_map(activation_map[:, _slice, :, :], args, slice_index=_slice)


def extract_activation_maps(args: ModelConfigBase) -> None:
    """
    Extracts and saves activation maps of a specific layer of a trained network
    :param args:
    :return:
    """
    model = args.create_model()
    if args.use_gpu:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()

    if args.test_start_epoch:
        checkpoint_path = args.get_path_to_checkpoint(args.test_start_epoch)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)  # type: ignore
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("Could not find checkpoint")

        model.eval()

        val_loader = args.create_data_loaders()[ModelExecutionMode.VAL]

        feature_extractor = model_hooks.HookBasedFeatureExtractor(model, layer_name=args.activation_map_layers)

        for batch, sample in enumerate(val_loader):

            sample = CroppedSample.from_dict(sample=sample)

            input_image = sample.image.cuda().float()

            feature_extractor(input_image)

            # access first image of batch of feature maps
            activation_map = feature_extractor.outputs[0][0].cpu().numpy()

            if len(activation_map.shape) == 4:
                visualize_3d_activation_map(activation_map, args)

            elif len(activation_map.shape) == 3:
                visualize_2d_activation_map(activation_map, args)

            else:
                raise NotImplementedError('cannot visualize activation map of shape', activation_map.shape)

            # Only visualize the first validation example
            break

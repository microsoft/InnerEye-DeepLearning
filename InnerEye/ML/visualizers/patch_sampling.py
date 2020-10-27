#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import param

from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.cropping_dataset import CroppingDataset
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.plotting import resize_and_save, scan_with_transparent_overlay
from InnerEye.ML.utils import augmentation, io_util, ml_util
from InnerEye.ML.utils.config_util import ModelConfigLoader
# The name of the folder inside the default outputs folder that will holds plots that show the effect of
# sampling random patches
from InnerEye.ML.utils.image_util import get_unit_image_header

PATCH_SAMPLING_FOLDER = "patch_sampling"


class CheckPatchSamplingConfig(GenericConfig):
    """
    Config class to store settings for patch sampling visualization script
    """
    model_name: str = param.String("Lung", doc="InnerEye model name e.g. Lung")
    local_dataset: str = param.String(None, doc="Path to the local dataset (e.g. dataset folder name)")
    output_folder: Path = param.ClassSelector(class_=Path, default=Path("patch_sampling_visualisations"),
                                              doc="Output folder where heatmaps and sampled images are saved")
    number_samples: int = param.Number(10, bounds=(1, None), doc="Number of images sampled")


def visualize_random_crops(sample: Sample,
                           config: SegmentationModelBase,
                           output_folder: Path) -> np.ndarray:
    """
    Simulate the effect of sampling random crops (as is done for trainig segmentation models), and store the results
    as a Nifti heatmap and as 3 axial/sagittal/coronal slices. The heatmap and the slices are stored in the given
    output folder, with filenames that contain the patient ID as the prefix.
    :param sample: The patient information from the dataset, with scans and ground truth labels.
    :param config: The model configuration.
    :param output_folder: The folder into which the heatmap and thumbnails should be written.
    :return: A numpy array that has the same size as the image, containing how often each voxel was contained in
    """
    output_folder.mkdir(exist_ok=True, parents=True)
    sample = CroppingDataset.create_possibly_padded_sample_for_cropping(
        sample=sample,
        crop_size=config.crop_size,
        padding_mode=config.padding_mode)
    logging.info(f"Processing sample: {sample.patient_id}")
    # Exhaustively sample with random crop function
    image_channel0 = sample.image[0]
    heatmap = np.zeros(image_channel0.shape, dtype=np.uint16)
    # Number of repeats should fit into the range of UInt16, because we will later save the heatmap as an integer
    # Nifti file of that datatype.
    repeats = 200
    for _ in range(repeats):
        slicers, _ = augmentation.slicers_for_random_crop(sample=sample,
                                                          crop_size=config.crop_size,
                                                          class_weights=config.class_weights)
        heatmap[slicers[0], slicers[1], slicers[2]] += 1
    is_3dim = heatmap.shape[0] > 1
    header = sample.metadata.image_header
    if not header:
        logging.warning(f"No image header found for patient {sample.patient_id}. Using default header.")
        header = get_unit_image_header()
    if is_3dim:
        ct_output_name = str(output_folder / f"{sample.patient_id}_ct.nii.gz")
        heatmap_output_name = str(output_folder / f"{sample.patient_id}_sampled_patches.nii.gz")
        io_util.store_as_nifti(image=heatmap,
                               header=header,
                               file_name=heatmap_output_name,
                               image_type=heatmap.dtype,
                               scale=False)
        io_util.store_as_nifti(image=image_channel0,
                               header=header,
                               file_name=ct_output_name,
                               image_type=sample.image.dtype,
                               scale=False)
    heatmap_scaled = heatmap.astype(dtype=np.float) / heatmap.max()
    # If the incoming image is effectively a 2D image with degenerate Z dimension, then only plot a single
    # axial thumbnail. Otherwise, plot thumbnails for all 3 dimensions.
    dimensions = list(range(3)) if is_3dim else [0]
    # Center the 3 thumbnails at one of the points where the heatmap attains a maximum. This should ensure that
    # the thumbnails are in an area where many of the organs of interest are located.
    max_heatmap_index = np.unravel_index(heatmap.argmax(), heatmap.shape) if is_3dim else (0, 0, 0)
    for dimension in dimensions:
        plt.clf()
        scan_with_transparent_overlay(scan=image_channel0,
                                      overlay=heatmap_scaled,
                                      dimension=dimension,
                                      position=max_heatmap_index[dimension] if is_3dim else 0,
                                      spacing=header.spacing)
        # Construct a filename that has a dimension suffix if we are generating 3 of them. For 2dim images, skip
        # the suffix.
        thumbnail = f"{sample.patient_id}_sampled_patches"
        if is_3dim:
            thumbnail += f"_dim{dimension}"
        thumbnail += ".png"
        resize_and_save(width_inch=5, height_inch=5, filename=output_folder / thumbnail)
    return heatmap


def visualize_random_crops_for_dataset(config: DeepLearningConfig,
                                       output_folder: Optional[Path] = None) -> None:
    """
    For segmentation models only: This function generates visualizations of the effect of sampling random patches
    for training. Visualizations are stored in both Nifti format, and as 3 PNG thumbnail files, in the output folder.
    :param config: The model configuration.
    :param output_folder: The folder in which the visualizations should be written. If not provided, use a subfolder
    "patch_sampling" in the models's default output folder
    """
    if not isinstance(config, SegmentationModelBase):
        return
    dataset_splits = config.get_dataset_splits()
    # Load a sample using the full image data loader
    full_image_dataset = FullImageDataset(config, dataset_splits.train)
    output_folder = output_folder or config.outputs_folder / PATCH_SAMPLING_FOLDER
    count = min(config.show_patch_sampling, len(full_image_dataset))
    for sample_index in range(count):
        sample = full_image_dataset.get_samples_at_index(index=sample_index)[0]
        visualize_random_crops(sample, config, output_folder=output_folder)


def main(args: CheckPatchSamplingConfig) -> None:
    # Identify paths to inputs and outputs
    commandline_args = {
        "train_batch_size": 1,
        "local_dataset": Path(args.local_dataset)
    }
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Create a config file
    config = ModelConfigLoader[SegmentationModelBase]().create_model_config_from_name(
        args.model_name, overrides=commandline_args)
    config.show_patch_sampling = args.number_samples
    ml_util.set_random_seed(config.random_seed)
    visualize_random_crops_for_dataset(config, output_folder=output_folder)


if __name__ == "__main__":
    main(CheckPatchSamplingConfig.parse_args())

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import param

from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.cropping_dataset import CroppingDataset
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.utils import augmentation, io_util, ml_util
from InnerEye.ML.utils.config_util import ModelConfigLoader

# The name of the folder inside the default outputs folder that will holds plots that show the effect of
# sampling random patches
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


def visualize_patch_sampling(sample: Sample,
                             config: SegmentationModelBase,
                             output_folder: Path) -> None:
    sample = CroppingDataset.create_possibly_padded_sample_for_cropping(
        sample=sample,
        crop_size=config.crop_size,
        padding_mode=config.padding_mode)
    print("Processing sample: ", sample.patient_id)

    # Exhaustively sample with random crop function
    image_channel0 = sample.image[0]
    heatmap = np.zeros(image_channel0.shape, dtype=np.uint16)
    for _ in range(1000):
        _, _, slicers = augmentation.random_crop(sample=sample,
                                                 crop_size=config.crop_size,
                                                 class_weights=config.class_weights)
        heatmap[slicers[0], slicers[1], slicers[2]] += 1
    ct_output_name = str(output_folder / f"{sample.patient_id}_ct.nii.gz")
    heatmap_output_name = str(output_folder / f"{sample.patient_id}_sampled_patches.nii.gz")
    if not sample.metadata.image_header:
        raise ValueError("Unable to save in Nifti format because no image header was found.")
    io_util.store_as_nifti(image=heatmap,
                           header=sample.metadata.image_header,
                           file_name=heatmap_output_name,
                           image_type=heatmap.dtype,
                           scale=False)
    io_util.store_as_nifti(image=image_channel0,
                           header=sample.metadata.image_header,
                           file_name=ct_output_name,
                           image_type=sample.image.dtype,
                           scale=False)


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

    # Set a random seed
    ml_util.set_random_seed(config.random_seed)

    # Get a dataloader object that checks csv
    dataset_splits = config.get_dataset_splits()

    # Load a sample using the full image data loader
    full_image_dataset = FullImageDataset(config, dataset_splits.train)
    for sample_index in range(args.number_samples):
        sample = full_image_dataset.get_samples_at_index(index=sample_index)[0]
        visualize_patch_sampling(sample, config, output_folder=args.output_folder)


if __name__ == "__main__":
    main(CheckPatchSamplingConfig.parse_args())

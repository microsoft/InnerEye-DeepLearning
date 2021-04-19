#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import param

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import ParserResult, create_runner_parser, parse_args_and_add_yaml_variables
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML import plotting
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import load_dataset_sources
from InnerEye.ML.deep_learning_config import ARGS_TXT
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.io_util import load_images_from_dataset_source


class NormalizeAndVisualizeConfig(GenericConfig):
    image_channel: Optional[str] = param.String(default=None,
                                                doc="The name of the image channel that should be normalized.")
    gt_channel: Optional[str] = param.String(default=None, doc="The name of the ground truth channel that should "
                                                               "be used when visualizing slices.")
    only_first: int = param.Integer(default=0,
                                    doc="Only process the first N images of the dataset, to speed up things.")
    result_folder: str = param.String(default="NormResults",
                                      doc="The folder to use to store the resulting plots. By default, "
                                          "plots will go into the 'NormResults' subfolder inside of the dataset "
                                          "folder. If a relative path is specified here, the folder will be created as"
                                          "a subfolder of the dataset folder. An absolute path can be used too.")
    ignore_mask: bool = param.Boolean(doc="If true, the mask channel specified in the image will not be used, and all "
                                          "histograms and normalization will use all image pixels.")


def create_parser(yaml_file_path: Path) -> ParserResult:
    """
    Create a parser for all runner arguments, even though we are only using a subset of the arguments.
    This way, we can get secrets handling in a consistent way.
    In particular, this will create arguments for
      --local_dataset
      --azure_dataset_id
    """
    parser = create_runner_parser(SegmentationModelBase)
    NormalizeAndVisualizeConfig.add_args(parser)
    return parse_args_and_add_yaml_variables(parser, yaml_config_file=yaml_file_path, fail_on_unknown_args=True)


def get_configs(default_model_config: SegmentationModelBase,
                yaml_file_path: Path) -> Tuple[SegmentationModelBase, AzureConfig, Dict]:
    parser_result = create_parser(yaml_file_path)
    args = parser_result.args
    runner_config = AzureConfig(**args)
    logging_to_stdout(args["log_level"])
    config = default_model_config or ModelConfigLoader().create_model_config_from_name(runner_config.model)
    config.apply_overrides(parser_result.overrides, should_validate=False)
    return config, runner_config, args


def main(yaml_file_path: Path) -> None:
    """
    Invoke either by
      * specifying a model, '--model Lung'
      * or specifying dataset and normalization parameters separately: --azure_dataset_id=foo --norm_method=None
    In addition, the arguments '--image_channel' and '--gt_channel' must be specified (see below).
    """
    config, runner_config, args = get_configs(SegmentationModelBase(should_validate=False), yaml_file_path)
    local_dataset = MLRunner(config, azure_config=runner_config).mount_or_download_dataset()
    assert local_dataset is not None
    dataframe = pd.read_csv(local_dataset / DATASET_CSV_FILE_NAME)
    normalizer_config = NormalizeAndVisualizeConfig(**args)
    actual_mask_channel = None if normalizer_config.ignore_mask else config.mask_id
    image_channel = normalizer_config.image_channel or config.image_channels[0]
    if not image_channel:
        raise ValueError("No image channel selected. Specify a model by name, or use the image_channel argument.")
    gt_channel = normalizer_config.gt_channel or config.ground_truth_ids[0]
    if not gt_channel:
        raise ValueError("No GT channel selected. Specify a model by name, or use the gt_channel argument.")

    dataset_sources = load_dataset_sources(dataframe,
                                           local_dataset_root_folder=local_dataset,
                                           image_channels=[image_channel],
                                           ground_truth_channels=[gt_channel],
                                           mask_channel=actual_mask_channel)
    result_folder = local_dataset
    if normalizer_config.result_folder is not None:
        result_folder = result_folder / normalizer_config.result_folder
    if not result_folder.is_dir():
        result_folder.mkdir()
    all_patient_ids = [*dataset_sources.keys()]
    if normalizer_config.only_first == 0:
        patient_ids_to_process = all_patient_ids
    else:
        patient_ids_to_process = all_patient_ids[:normalizer_config.only_first]
    args_file = result_folder / ARGS_TXT
    args_file.write_text(" ".join(sys.argv[1:]))
    config_file = result_folder / "config.txt"
    config_file.write_text(str(config))
    normalizer = PhotometricNormalization(config)
    for patient_id in patient_ids_to_process:
        print(f"Starting to process patient {patient_id}")
        images = load_images_from_dataset_source(dataset_sources[patient_id])
        plotting.plot_normalization_result(images, normalizer, result_folder, result_prefix=image_channel)


if __name__ == '__main__':
    main(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE)

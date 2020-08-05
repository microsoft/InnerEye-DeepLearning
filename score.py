#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
IMPORTANT CONSTRAINTS
---------------------
1) DO NOT move this file, this file is expected to be in the root directory of this project by
the caller code in AML

This is an executable that is called by the python_wrapper.py which handles inference of a single image
(which is an AML construct called scoring, thus the name:
https://docs.microsoft.com/en-us/python/api/overview/azureml-sdk/?view=azure-ml-py).
"""

import logging
import os
import sys
from distutils.dir_util import copy_tree
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import param
from azureml.core import Run

from InnerEye.Azure.azure_util import is_offline_run_context
from InnerEye.Common import fixed_paths
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import FullImageInferencePipelineBase, InferencePipeline
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.io_util import ImageWithHeader, load_nifti_image, reverse_tuple_float3, store_as_ubyte_nifti
from run_scoring import PYTHONPATH_ENVIRONMENT_VARIABLE_NAME

DEFAULT_DATA_FOLDER = "data"
DEFAULT_TEST_IMAGE_NAME = "test.nii.gz"


class ScorePipelineConfig(GenericConfig):
    data_root: str = param.String(None, doc="Path to the folder that contains the data "
                                            "(image channels and checkpoints) for scoring.")
    project_root: str = param.String(None, doc="Path to the folder that contains code root.")
    test_image_channels: List[str] = param.List([DEFAULT_TEST_IMAGE_NAME], class_=str, instantiate=False,
                                                bounds=(1, None),
                                                doc="The name of the image channels to run the pipeline on.")
    result_image_name: str = param.String(DEFAULT_RESULT_IMAGE_NAME, doc="The name of the resulting image from the "
                                                                         "pipeline.")
    use_gpu: bool = param.Boolean(True, doc="If GPU should be used or not.")


def init_from_model_inference_json(model_path: Path, use_gpu: bool = True) -> Tuple[FullImageInferencePipelineBase,
                                                                                      SegmentationModelBase]:
    """
    Loads the config and inference pipeline from the current directory using fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
    :return: Tuple[InferencePipeline, Config]
    """
    logging.info('Python version: ' + sys.version)
    path_to_model_inference_config = model_path / fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
    logging.info(f'path_to_model_inference_config: {path_to_model_inference_config}')
    model_inference_config = read_model_inference_config(str(path_to_model_inference_config))
    logging.info(f'model_inference_config: {model_inference_config}')
    full_path_to_checkpoints = [model_path / x for x in model_inference_config.checkpoint_paths]
    logging.info(f'full_path_to_checkpoints: {full_path_to_checkpoints}')
    loader = ModelConfigLoader[SegmentationModelBase](
        model_configs_namespace=model_inference_config.model_configs_namespace)
    model_config = loader.create_model_config_from_name(model_name=model_inference_config.model_name)
    return create_inference_pipeline(model_config, full_path_to_checkpoints, use_gpu)


def create_inference_pipeline(model_config: SegmentationModelBase,
                              full_path_to_checkpoints: List[Path], use_gpu: bool = True) \
        -> Tuple[FullImageInferencePipelineBase, SegmentationModelBase]:
    """
    Create pipeline for inference, this can be a single model inference pipeline or an ensemble, if multiple
    checkpoints provided.
    :param model_config: Model config to use to create the pipeline.
    :param full_path_to_checkpoints: Checkpoints to use for model inference.
    :param use_gpu: If GPU should be used or not.
    """
    model_config.use_gpu = use_gpu
    logging.info('test_config: ' + model_config.model_name)

    inference_pipeline: Optional[FullImageInferencePipelineBase]
    if len(full_path_to_checkpoints) == 1:
        inference_pipeline = InferencePipeline.create_from_checkpoint(
            path_to_checkpoint=full_path_to_checkpoints[0],
            model_config=model_config)
    else:
        inference_pipeline = EnsemblePipeline.create_from_checkpoints(path_to_checkpoints=full_path_to_checkpoints,
                                                                      model_config=model_config)
    if inference_pipeline is None:
        raise ValueError("Cannot create inference pipeline")

    return inference_pipeline, model_config


def read_model_inference_config(path_to_model_inference_config: str) -> ModelInferenceConfig:
    with open(path_to_model_inference_config, 'r', encoding='utf-8') as file:
        model_inference_config = ModelInferenceConfig.from_json(file.read())  # type: ignore
    return model_inference_config


def is_spacing_valid(spacing: TupleFloat3, dataset_expected_spacing_xyz: TupleFloat3) -> bool:
    absolute_tolerance = 1e-1
    return np.allclose(spacing, dataset_expected_spacing_xyz, atol=absolute_tolerance)


def run_inference(images_with_header: List[ImageWithHeader],
                  inference_pipeline: FullImageInferencePipelineBase,
                  config: SegmentationModelBase) -> np.ndarray:
    """
    Runs inference on a list of channels and given a config and inference pipeline
    :param images_with_header:
    :param inference_pipeline:
    :param config:
    :return: segmentation
    """
    # Check the image has the correct spacing
    if config.dataset_expected_spacing_xyz:
        for image_with_header in images_with_header:
            spacing_xyz = reverse_tuple_float3(image_with_header.header.spacing)
            if not is_spacing_valid(spacing_xyz, config.dataset_expected_spacing_xyz):
                raise ValueError(f'Input image has spacing {spacing_xyz} '
                                 f'but expected {config.dataset_expected_spacing_xyz}')
    # Photo norm
    photo_norm = PhotometricNormalization(config_args=config)
    photo_norm_images = [photo_norm.transform(image_with_header.image) for image_with_header in images_with_header]
    segmentation = inference_pipeline.predict_and_post_process_whole_image(
        image_channels=np.array(photo_norm_images),
        voxel_spacing_mm=images_with_header[0].header.spacing
    ).segmentation

    return segmentation


def score_image(args: ScorePipelineConfig) -> Path:
    """
    Perform model inference on a single image. By doing the following:
    1) Copy the provided data root directory to the root (this contains the model checkpoints and image to infer)
    2) Instantiate an inference pipeline based on the provided model_inference.json in the snapshot
    3) Store the segmentation file in the current directory
    4) Upload the segmentation to AML
    :param args:
    :return:
    """
    logging.getLogger().setLevel(logging.INFO)
    project_root = Path(args.project_root)

    # copy the model to the current directory
    copy_tree(args.data_root, str(project_root))
    logging.info(f'Copied contents of data_root: {args.data_root} to {project_root}')

    run_context = Run.get_context()
    logging.info(f"Run context={run_context.id}")

    images = [load_nifti_image(project_root / DEFAULT_DATA_FOLDER / x) for x in args.test_image_channels]
    inference_pipeline, config = init_from_model_inference_json(project_root, args.use_gpu)
    segmentation = run_inference(images, inference_pipeline, config)

    segmentation_file_name = str(project_root / args.result_image_name)
    result_dst = store_as_ubyte_nifti(segmentation, images[0].header, segmentation_file_name)
    if not is_offline_run_context(run_context):
        run_context.upload_file(args.result_image_name, segmentation_file_name)
    logging.info(f"Segmentation completed: {result_dst}")

    return Path(result_dst)


def main():
    print(f"{PYTHONPATH_ENVIRONMENT_VARIABLE_NAME}: {os.environ.get(PYTHONPATH_ENVIRONMENT_VARIABLE_NAME)}")
    score_image(ScorePipelineConfig.parse_args())


if __name__ == "__main__":
    main()

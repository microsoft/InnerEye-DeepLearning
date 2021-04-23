#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from collections import defaultdict
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import zipfile

import numpy as np
import param
from azureml.core import Run
from InnerEye_DICOM_RT.nifti_to_dicom_rt_converter import rtconvert

from InnerEye.Azure.azure_util import is_offline_run_context
from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_ZIP_DICOM_NAME
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.type_annotations import TupleFloat3, TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_inference_config import ModelInferenceConfig
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import FullImageInferencePipelineBase, InferencePipeline
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.io_util import ImageWithHeader, load_nifti_image, reverse_tuple_float3, store_as_ubyte_nifti, \
    load_dicom_series_and_save


class ScorePipelineConfig(GenericConfig):
    data_folder: Path = param.ClassSelector(class_=Path, default=Path.cwd(),
                                            doc="Path to the folder that contains the images that should be scored")
    model_folder: str = param.String(doc="Path to the folder that contains the model, in particular inference "
                                         "configuration and checkpoints. Defaults to the folder where the current "
                                         "file lives.")
    image_files: List[str] = param.List([fixed_paths.DEFAULT_TEST_IMAGE_NAME], class_=str, instantiate=False,
                                        bounds=(1, None),
                                        doc="The name of the images channels to run the pipeline on. These "
                                            "files must exist in the data_folder.")
    result_image_name: str = param.String(DEFAULT_RESULT_IMAGE_NAME,
                                          doc="The name of the result image, created in the project root folder.")
    use_gpu: bool = param.Boolean(True, doc="If GPU should be used or not.")
    use_dicom: bool = param.Boolean(False, doc="If images to be scored are DICOM and output to be DICOM-RT. "
                                               "If this is set then image_files should contain a single zip file "
                                               "containing a set of DICOM files.")
    result_zip_dicom_name: str = param.String(DEFAULT_RESULT_ZIP_DICOM_NAME,
                                              doc="The name of the zipped DICOM-RT file if use_dicom set.")
    model_id: str = param.String(allow_None=False,
                                 doc="The AzureML model ID. This is added to the SoftwareVersions DICOM tag in the "
                                     "DICOM-RT output")


def init_from_model_inference_json(model_folder: Path, use_gpu: bool = True) -> Tuple[FullImageInferencePipelineBase,
                                                                                      SegmentationModelBase]:
    """
    Loads the config and inference pipeline from the current directory using fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
    :return: Tuple[InferencePipeline, Config]
    """
    logging.info('Python version: ' + sys.version)
    path_to_model_inference_config = model_folder / fixed_paths.MODEL_INFERENCE_JSON_FILE_NAME
    logging.info(f'path_to_model_inference_config: {path_to_model_inference_config}')
    model_inference_config = read_model_inference_config(str(path_to_model_inference_config))
    logging.info(f'model_inference_config: {model_inference_config}')
    full_path_to_checkpoints = [model_folder / x for x in model_inference_config.checkpoint_paths]
    logging.info(f'full_path_to_checkpoints: {full_path_to_checkpoints}')
    loader = ModelConfigLoader(model_configs_namespace=model_inference_config.model_configs_namespace)
    model_config = loader.create_model_config_from_name(model_name=model_inference_config.model_name)
    return create_inference_pipeline(model_config, full_path_to_checkpoints, use_gpu)


def create_inference_pipeline(model_config: SegmentationModelBase,
                              full_path_to_checkpoints: List[Path],
                              use_gpu: bool = True) \
        -> Tuple[FullImageInferencePipelineBase, SegmentationModelBase]:
    """
    Create pipeline for inference, this can be a single model inference pipeline or an ensemble, if multiple
    checkpoints provided.
    :param model_config: Model config to use to create the pipeline.
    :param full_path_to_checkpoints: Checkpoints to use for model inference.
    :param use_gpu: If GPU should be used or not.
    """
    model_config.max_num_gpus = -1 if use_gpu else 0
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


def extract_zipped_files_and_flatten(zip_file_path: Path, extraction_folder: Path) -> None:
    """
    Unzip a zip file and extract all the files discarding any folders they
    may have in the zip file.

    :param zip_file_path: Path to zip file.
    :param extraction_folder: Path to extraction folder.
    """
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zipinfos_by_name = defaultdict(list)
        for zipped_file in zip_file.infolist():
            if not zipped_file.is_dir():
                # discard the path, if any, to just get the filename and suffix
                name = os.path.basename(zipped_file.filename)
                zipinfos_by_name[name].append(zipped_file)
        duplicates = {name: zipinfos for name, zipinfos in zipinfos_by_name.items() if len(zipinfos) > 1}
        if len(duplicates) > 0:
            warnings = ""
            for name, zipinfos in duplicates.items():
                joint_paths = ", ".join([os.path.dirname(zipinfo.filename) for zipinfo in zipinfos])
                warnings += f"File {name} is duplicated in folders {joint_paths}.\n"
            raise ValueError("Zip file contains duplicates.\n" + warnings)
        for name, zipinfos in zipinfos_by_name.items():
            zipinfo = zipinfos[0]
            zipinfo.filename = name
            zip_file.extract(zipinfo, str(extraction_folder))


def convert_zipped_dicom_to_nifti(zip_file_path: Path, reference_series_folder: Path,
                                  nifti_file_path: Path) -> None:
    """
    Given a zip file, extract DICOM series and convert to Nifti format.

    This function:
    1) Unzips the file at zip_file_path into reference_series_folder,
    assumed to contain a DICOM series.
    2) Creates a Nifti file from the DICOM series.

    :param zip_file_path: Path to a zip file.
    :param reference_series_folder: Folder to unzip DICOM series into.
    :param nifti_file_path: Path to target Nifti file.
    """
    extract_zipped_files_and_flatten(zip_file_path, reference_series_folder)
    load_dicom_series_and_save(reference_series_folder, nifti_file_path)


def convert_rgb_colour_to_hex(colour: TupleInt3) -> str:
    """
    Config colours are stored as TupleInt3's, but DICOM-RT convert expects
    hexadecimal strings. This function converts them into the correct
    format.

    :param colour: RGB colour as a TupleInt3.
    :return: Colour formatted as a hex string.
    """
    return '{0:02X}{1:02X}{2:02X}'.format(colour[0], colour[1], colour[2])


def convert_nifti_to_zipped_dicom_rt(nifti_file: Path, reference_series: Path, scratch_folder: Path,
                                     config: SegmentationModelBase, dicom_rt_zip_file_name: str, model_id: str) -> Path:
    """
    Given a Nifti file and a reference DICOM series, create zip file containing a DICOM-RT file.

    Calls rtconvert with the given Nifti file, reference DICOM series and configuration from
    config to create a DICOM-RT file in the scratch folder. This is then zipped and a path to
    the zip returned.

    :param nifti_file: Path to Nifti file.
    :param reference_series: Path to folder containing reference DICOM series.
    :param scratch_folder: Scratch folder to extract files into.
    :param config: Model config.
    :param dicom_rt_zip_file_name: Target DICOM-RT zip file name, ending in .dcm.zip.
    :param model_id: The AzureML model ID <model_name>:<ID>
    :return: Path to DICOM-RT file.
    """
    dicom_rt_file_path = scratch_folder / Path(dicom_rt_zip_file_name).with_suffix("")
    (stdout, stderr) = rtconvert(
        in_file=nifti_file,
        reference_series=reference_series,
        out_file=dicom_rt_file_path,
        struct_names=config.ground_truth_ids_display_names,
        struct_colors=[convert_rgb_colour_to_hex(rgb) for rgb in config.colours],
        fill_holes=config.fill_holes,
        roi_interpreted_types=config.roi_interpreted_types,
        manufacturer=config.manufacturer,
        interpreter=config.interpreter,
        modelId=model_id
    )
    # Log stdout, stderr from DICOM-RT conversion.
    logging.debug("stdout: %s", stdout)
    logging.debug("stderr: %s", stderr)
    dicom_rt_zip_file_path = scratch_folder / dicom_rt_zip_file_name
    with zipfile.ZipFile(dicom_rt_zip_file_path, 'w') as dicom_rt_zip:
        dicom_rt_zip.write(dicom_rt_file_path, dicom_rt_file_path.name)
    return dicom_rt_zip_file_path


def check_input_file(data_folder: Path, filename: str) -> Path:
    """
    Check the folder: data_folder contains a file with name: filename.

    If the file does not exist then raise a FileNotFoundError exception. Otherwise return the
    path to the file.

    :param data_folder: Path to data folder.
    :param filename: Filename within data folder to test.
    :return: Full path to filename.
    """
    full_file_path = data_folder / filename
    if not full_file_path.exists():
        message = \
            str(data_folder) if data_folder.is_absolute() else f"{data_folder}, absolute: {data_folder.absolute()}"
        raise FileNotFoundError(f"File {filename} does not exist in data folder {message}")
    return full_file_path


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
    score_py_folder = Path(__file__).parent
    model_folder = Path(args.model_folder or str(score_py_folder))

    run_context = Run.get_context()
    logging.info(f"Run context={run_context.id}")

    if args.use_dicom:
        # Only a single zip file is supported.
        if len(args.image_files) > 1:
            raise ValueError("Supply exactly one zip file in args.images.")
        input_zip_file = check_input_file(args.data_folder, args.image_files[0])
        reference_series_folder = model_folder / "temp_extraction"
        nifti_filename = model_folder / "temp_nifti.nii.gz"
        convert_zipped_dicom_to_nifti(input_zip_file, reference_series_folder, nifti_filename)
        test_images = [nifti_filename]
    else:
        test_images = [check_input_file(args.data_folder, file) for file in args.image_files]

    images = [load_nifti_image(file) for file in test_images]

    inference_pipeline, config = init_from_model_inference_json(model_folder, args.use_gpu)
    segmentation = run_inference(images, inference_pipeline, config)

    segmentation_file_name = model_folder / args.result_image_name
    result_dst = store_as_ubyte_nifti(segmentation, images[0].header, segmentation_file_name)

    if args.use_dicom:
        result_dst = convert_nifti_to_zipped_dicom_rt(result_dst, reference_series_folder, model_folder,
                                                      config, args.result_zip_dicom_name, args.model_id)

    if not is_offline_run_context(run_context):
        upload_file_name = args.result_zip_dicom_name if args.use_dicom else args.result_image_name
        run_context.upload_file(upload_file_name, str(result_dst))
    logging.info(f"Segmentation completed: {result_dst}")
    return result_dst


def main() -> None:
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    score_image(ScorePipelineConfig.parse_args())


if __name__ == "__main__":
    main()

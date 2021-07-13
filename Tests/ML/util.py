#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from azureml.core import Workspace

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import PathOrString, TupleInt3
from InnerEye.ML.dataset.full_image_dataset import PatientDatasetSource
from InnerEye.ML.dataset.sample import PatientMetadata, Sample
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.lightning_loggers import StoringLogger
from InnerEye.ML.model_training import model_train
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.runner import Runner
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.io_util import ImageHeader, ImageWithHeader
from InnerEye.ML.utils.ml_util import is_gpu_available

TEST_CHANNEL_IDS = ["channel1", "channel2"]
TEST_MASK_ID = "mask"
TEST_GT_ID = "region"

machine_has_gpu = is_gpu_available()
no_gpu_available = not machine_has_gpu


def create_dataset_csv_file(csv_string: str, dst: Path) -> Path:
    """Creates a dataset.csv in the destination path from the csv_string provided"""
    (dst / "dataset.csv").write_text(csv_string)
    return Path(dst)


def content_mismatch(actual: Any, expected: Any) -> str:
    """Returns error message for content mismatch."""
    return "Content mismatch. \nActual:\n {}\nExpected:\n {}".format(actual, expected)


def get_nifti_shape(full_path: PathOrString) -> TupleInt3:
    """Returns the size of the image in the given Nifti file, as an (X, Y, Z) tuple."""
    image_with_header = io_util.load_nifti_image(full_path)
    return get_image_shape(image_with_header)


def get_image_shape(image_with_header: ImageWithHeader) -> TupleInt3:
    return image_with_header.image.shape[0], image_with_header.image.shape[1], image_with_header.image.shape[2]


def load_train_and_test_data_channels(patient_ids: List[int],
                                      normalization_fn: PhotometricNormalization) -> List[Sample]:
    if np.any(np.asarray(patient_ids) <= 0):
        raise ValueError("data_items must be >= 0")

    file_name = lambda k, y: full_ml_test_data_path("train_and_test_data") / f"id{k}_{y}.nii.gz"

    get_sample = lambda z: io_util.load_images_from_dataset_source(dataset_source=PatientDatasetSource(
        metadata=PatientMetadata(patient_id=z),
        image_channels=[file_name(z, c) for c in TEST_CHANNEL_IDS],
        mask_channel=file_name(z, TEST_MASK_ID),
        ground_truth_channels=[file_name(z, TEST_GT_ID)],
        allow_incomplete_labels=False
    ))

    samples = []
    for x in patient_ids:
        sample = get_sample(x)
        sample = Sample(image=normalization_fn.transform(sample.image, sample.mask),
                        mask=sample.mask,
                        labels=sample.labels,
                        metadata=sample.metadata)
        samples.append(sample)

    return samples


def assert_file_contains_string(full_file: Union[str, Path], expected: Any = None) -> None:
    """
    Checks if the given file contains an expected string
    :param full_file: The path to the file.
    :param expected: The expected contents of the file, as a string.
    """
    logging.info("Checking file {}".format(full_file))
    file_path = full_file if isinstance(full_file, Path) else Path(full_file)
    assert_file_exists(file_path)
    if expected is not None:
        assert expected.strip() in file_path.read_text()


def assert_text_files_match(full_file: Path, expected_file: Path) -> None:
    """
    Checks line by line (ignoring leading and trailing spaces) if the given two files contains the exact same strings
    :param full_file: The path to the file.
    :param expected_file: The expected file.
    """
    with full_file.open() as f1, expected_file.open() as f2:
        for line1, line2 in zip(f1, f2):
            _assert_line(line1, line2)


def _assert_line(actual: str, expected: str) -> None:
    actual = actual.strip()
    expected = expected.strip()
    assert actual == expected, content_mismatch(actual, expected)


def assert_file_exists(file_path: Path) -> None:
    """
    Checks if the given file exists.
    """
    assert file_path.exists(), f"File does not exist: {file_path}"


def assert_nifti_content(full_file: PathOrString,
                         expected_shape: TupleInt3,
                         expected_header: ImageHeader,
                         expected_values: List[int],
                         expected_type: type) -> None:
    """
    Checks if the given nifti file contains the expected unique values, and has the expected type and shape.
    :param full_file: The path to the file.
    :param expected_shape: The expected shape of the image in the nifti file.
    :param expected_header: the expected image header
    :param expected_values: The expected unique values in the image array.
    :param expected_type: The expected type of the stored values.
    """
    if isinstance(full_file, str):
        full_file = Path(full_file)
    assert_file_exists(full_file)
    image_with_header = io_util.load_nifti_image(full_file, None)
    assert image_with_header.image.shape == expected_shape, content_mismatch(image_with_header.image.shape,
                                                                             expected_shape)
    assert image_with_header.image.dtype == np.dtype(expected_type), content_mismatch(image_with_header.image.dtype,
                                                                                      expected_type)
    image = np.unique(image_with_header.image).tolist()
    assert image == expected_values, content_mismatch(image, expected_values)
    assert image_with_header.header == expected_header


def assert_tensors_equal(t1: torch.Tensor, t2: Union[torch.Tensor, List], abs: float = 1e-6) -> None:
    """
    Checks if the shapes of the given tensors is equal, and the values are approximately equal, with a given
    absolute tolerance.
    """
    if isinstance(t2, list):
        t2 = torch.tensor(t2)
    assert t1.shape == t2.shape, "Shapes must match"
    # Alternative is to use torch.allclose here, but that method also checks that datatypes match. This makes
    # writing the test cases more cumbersome.
    v1 = t1.flatten().tolist()
    v2 = t2.flatten().tolist()
    assert v1 == pytest.approx(v2, abs=abs), f"Tensor elements don't match with tolerance {abs}: {v1} != {v2}"


def assert_binary_files_match(actual_file: Path, expected_file: Path) -> None:
    """
    Checks if two files contain exactly the same bytes. If PNG files mismatch, additional diagnostics is printed.
    """
    # Uncomment this line to batch-update all result files that use this assert function
    # expected_file.write_bytes(actual_file.read_bytes())
    assert_file_exists(actual_file)
    assert_file_exists(expected_file)
    actual = actual_file.read_bytes()
    expected = expected_file.read_bytes()
    if actual == expected:
        return
    if actual_file.suffix == ".png" and expected_file.suffix == ".png":
        actual_image = Image.open(actual_file)
        expected_image = Image.open(expected_file)
        actual_size = actual_image.size
        expected_size = expected_image.size
        assert actual_size == expected_size, f"Image sizes don't match: actual {actual_size}, expected {expected_size}"
        assert np.allclose(np.array(actual_image), np.array(expected_image)), "Image pixel data does not match."
    assert False, f"File contents does not match: len(actual)={len(actual)}, len(expected)={len(expected)}"


def csv_column_contains_value(
        csv_file_path: Path,
        column_name: str,
        value: Any,
        contains_only_value: bool = True) -> bool:
    """
    Checks that the column in the csv file contains the given value (and perhaps only contains that value)
    :param csv_file_path: The path to the CSV
    :param column_name: The name of the column in which we look for the value
    :param value: The value to look for
    :param contains_only_value: Check that this is the only value in the column (default True)
    :returns: Boolean, whether the CSV column contains the value (and perhaps only the value)
    """
    result = True
    if not csv_file_path.exists:
        raise ValueError(f"The CSV at {csv_file_path} does not exist.")
    df = pd.read_csv(csv_file_path)
    if column_name not in df.columns:
        ValueError(f"The column {column_name} is not in the CSV at {csv_file_path}, which has columns {df.columns}.")
    if value:
        result = result and value in df[column_name].unique()
    else:
        result = result and df[column_name].isnull().any()
    if contains_only_value:
        if value:
            result = result and df[column_name].nunique(dropna=True) == 1
        else:
            result = result and df[column_name].nunique(dropna=True) == 0
    return result


DummyPatientMetadata = PatientMetadata(patient_id='42')


def get_model_loader(namespace: Optional[str] = None) -> ModelConfigLoader:
    """
    Returns a ModelConfigLoader for segmentation models, with the given non-default namespace (if not None)
    to search under.
    """
    return ModelConfigLoader(model_configs_namespace=namespace)


def get_default_azure_config() -> AzureConfig:
    """
    Gets the Azure-related configuration options, using the default settings file settings.yaml.
    """
    return AzureConfig.from_yaml(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE,
                                 project_root=fixed_paths.repository_root_directory())


def get_default_checkpoint_handler(model_config: DeepLearningConfig, project_root: Path) -> CheckpointHandler:
    """
    Gets a checkpoint handler, using the given model config and the default azure configuration.
    """
    azure_config = get_default_azure_config()
    lightning_container = InnerEyeContainer(model_config)
    return CheckpointHandler(azure_config=azure_config,
                             container=lightning_container,
                             project_root=project_root)


def get_default_workspace() -> Workspace:
    """
    Gets the project's default AzureML workspace.
    :return:
    """
    return get_default_azure_config().get_workspace()


def model_train_unittest(config: Optional[DeepLearningConfig],
                         dirs: OutputFolderForTests,
                         checkpoint_handler: Optional[CheckpointHandler] = None,
                         lightning_container: Optional[LightningContainer] = None) -> \
        Tuple[StoringLogger, CheckpointHandler]:
    """
    A shortcut for running model training in the unit test suite. It runs training for the given config, with the
    default checkpoint handler initialized to point to the test output folder specified in dirs.
    :param config: The configuration of the model to train.
    :param dirs: The test fixture that provides an output folder for the test.
    :param lightning_container: An optional LightningContainer object that will be pass through to the training routine.
    :param checkpoint_handler: The checkpoint handler that should be used for training. If not provided, it will be
    created via get_default_checkpoint_handler.
    :return: Tuple[StoringLogger, CheckpointHandler]
    """
    runner = MLRunner(model_config=config, container=lightning_container)
    # Setup will set random seeds before model creation, and set the model in the container.
    # It will also set random seeds correctly. Later we use so initialized container.
    # For all tests running in AzureML, we need to skip the downloading of datasets that would otherwise happen,
    # because all unit test configs come with their own local dataset already.
    runner.setup(use_mount_or_download_dataset=False)
    if checkpoint_handler is None:
        azure_config = get_default_azure_config()
        checkpoint_handler = CheckpointHandler(azure_config=azure_config,
                                               container=runner.container,
                                               project_root=dirs.root_dir)
    _, storing_logger = model_train(checkpoint_path=checkpoint_handler.get_recovery_or_checkpoint_path_train(),
                                    container=runner.container)
    checkpoint_handler.additional_training_done()
    return storing_logger, checkpoint_handler  # type: ignore


def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=fixed_paths.repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


model_loader_including_tests = get_model_loader(namespace="Tests.ML.configs")

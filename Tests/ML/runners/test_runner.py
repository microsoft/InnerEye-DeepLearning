#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import shutil
import time
from pathlib import Path
from typing import List, Optional, Tuple
from unittest import mock
from unittest.mock import Mock

import numpy as np
import pytest
from azureml.core.run import Run
from azureml.train.hyperdrive.runconfig import HyperDriveConfig

from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, RUN_RECOVERY_ID_KEY_NAME
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.common_util import SUBJECT_METRICS_FILE_NAME, ModelProcessing, get_best_epoch_results_path
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.common import LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.configs.unit_testing.passthrough_model import PassThroughModel
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.metrics import InferenceMetricsForSegmentation
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.runner import Runner
from InnerEye.ML.utils import io_util
from InnerEye.ML.visualizers.plot_cross_validation import RUN_RECOVERY_ID_KEY, PlotCrossValidationConfig
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.runners.test_compare_against_baselines import create_dataset_df, create_metrics_df
from Tests.ML.util import get_default_checkpoint_handler
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


def create_smaller_image(image_size: TupleInt3, source_image_dir: Path, target_image_dir: Path,
                         image_file_name: str) -> None:
    """
    Load an image from source_image_dir and create another random image in target_image_dir with same header and
    target size.

    :param image_size: Target image size.
    :param source_image_dir: Source image directory.
    :param target_image_dir: Target image directory.
    :param image_file_name: Common image file name.
    :return: None.
    """
    source_image = io_util.load_nifti_image(source_image_dir / image_file_name)
    source_image_data = source_image.image
    min_data_val = np.min(source_image_data)
    max_data_val = np.max(source_image_data)

    image = np.random.randint(low=min_data_val, high=max_data_val + 1, size=image_size)
    io_util.store_as_nifti(image, source_image.header, target_image_dir / image_file_name, np.short)


def create_train_and_test_data_small(image_size: TupleInt3, source_image_dir: Path,
                                     target_image_dir: Path) -> None:
    """
    Create smaller, random, versions of the images from source_image_dir in target_image_dir.

    :param image_size: Target image size:
    :param source_image_dir: Source image directory.
    :param target_image_dir: Target image directory.
    :return: None.
    """
    for channel_file_name in ["id1_channel1.nii.gz", "id1_channel2.nii.gz",
                              "id2_channel1.nii.gz", "id2_channel2.nii.gz"]:
        create_smaller_image(image_size, source_image_dir, target_image_dir, channel_file_name)

    for mask_file_name in ["id1_mask.nii.gz", "id2_mask.nii.gz"]:
        create_smaller_image(image_size, source_image_dir, target_image_dir, mask_file_name)

    for region_file_name in ["id1_region.nii.gz", "id2_region.nii.gz"]:
        create_smaller_image(image_size, source_image_dir, target_image_dir, region_file_name)


def create_train_and_test_data_small_dataset(image_size: TupleInt3,
                                             source_dir: Path, source_images_folder: str,
                                             target_dir: Path, target_images_folder: str) -> Path:
    """
    Create smaller, random, versions of the dataset and images from source_dir in target_dir.

    :param image_size: Target image size:
    :param source_dir: Source dataset directory.
    :param source_images_folder: Source images folder.
    :param target_dir: Target dataset directory.
    :param target_images_folder: Target images folder.
    :return: target_dir.
    """
    # Load and rewrite dataset.csv
    csv_str = (source_dir / "dataset.csv").read_text()
    csv_str = csv_str.replace(source_images_folder, target_images_folder)

    target_dir.mkdir(parents=True)
    (target_dir / "dataset.csv").write_text(csv_str)

    source_image_dir = source_dir / source_images_folder

    target_image_dir = target_dir / target_images_folder
    target_image_dir.mkdir()
    create_train_and_test_data_small(image_size, source_image_dir, target_image_dir)
    return target_dir


def create_mock_run(mock_upload_path: Path, config: DeepLearningConfig) -> Run:
    """
    Create a mock AzureML Run object.

    :param mock_upload_path: Path to folder to store uploaded folders.
    :param config: Deep learning config.
    :return: Mock Run.
    """

    def mock_upload_folder(name: str, path: str, datastore_name: str = None) -> None:
        """
        Mock AzureML function Run.upload_folder.
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py#upload-folder-name--path--datastore-name-none-
        """
        shutil.copytree(src=path, dst=mock_upload_path / name)

    def mock_download_file(name: str, output_file_path: str = None, _validate_checksum: bool = False) -> None:
        """
        Mock AzureML function Run.download_file.
        https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.run(class)?view=azure-ml-py#download-file-name--output-file-path-none---validate-checksum-false-
        """
        if output_file_path is not None:
            src = mock_upload_path / name
            if src.name == DATASET_CSV_FILE_NAME:
                dataset_df = create_dataset_df()
                dataset_df.to_csv(output_file_path)
            elif src.name == SUBJECT_METRICS_FILE_NAME:
                metrics_df = create_metrics_df()
                metrics_df.to_csv(output_file_path)

    child_runs: List[Run] = []
    for i in range(config.number_of_cross_validation_splits):
        child_run = Mock(name=f'mock_child_run{i}')
        child_run.__class__ = Run
        child_run.download_file = Mock(name='mock_download_file', side_effect=mock_download_file)
        child_run.id = f'child_id:{i}'
        child_run.get_tags = lambda: {CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: i, RUN_RECOVERY_ID_KEY: 'rec_id'}
        child_runs.append(child_run)

    run = Mock(name='mock_run')
    run.__class__ = Run
    run.download_file = Mock(name='mock_download_file', side_effect=mock_download_file)
    run.get_children.return_value = child_runs
    run.get_tags = lambda: {RUN_RECOVERY_ID_KEY: 'rec_id'}
    run.id = 'run_id:1'
    run.tags = {RUN_RECOVERY_ID_KEY_NAME: 'id'}
    run.upload_folder = Mock(name='mock_upload_folder', side_effect=mock_upload_folder)

    return run


@pytest.mark.parametrize("perform_cross_validation", [True, False])
def test_model_inference_train_and_test_default(test_output_dirs: OutputFolderForTests,
                                                perform_cross_validation: bool) -> None:
    """
    Test inference defaults with ModelProcessing.DEFAULT.

    :param test_output_dirs: Test output directories.
    :param perform_cross_validation: Whether to test with cross validation.
    :return: None.
    """
    run_model_inference_train_and_test(test_output_dirs,
                                       perform_cross_validation,
                                       model_proc=ModelProcessing.DEFAULT)


@pytest.mark.parametrize("perform_cross_validation", [True, False])
@pytest.mark.parametrize("inference_on_set", [(True, False, False), (False, True, False), (False, False, True)])
def test_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                        perform_cross_validation: bool,
                                        inference_on_set: Tuple[bool, bool, bool]) -> None:
    """
    Test inference overrides with ModelProcessing.DEFAULT.

    :param test_output_dirs: Test output directories.
    :param perform_cross_validation: Whether to test with cross validation.
    :param inference_on_set: Overrides for inference on data sets.
    :return: None.
    """
    (inference_on_train_set, inference_on_val_set, inference_on_test_set) = inference_on_set
    run_model_inference_train_and_test(test_output_dirs,
                                       perform_cross_validation,
                                       inference_on_train_set=inference_on_train_set,
                                       inference_on_val_set=inference_on_val_set,
                                       inference_on_test_set=inference_on_test_set,
                                       model_proc=ModelProcessing.DEFAULT)


def test_ensemble_model_inference_train_and_test_default(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test inference defaults with ModelProcessing.ENSEMBLE_CREATION.

    :param test_output_dirs: Test output directories.
    :return: None.
    """
    run_model_inference_train_and_test(test_output_dirs,
                                       True,
                                       model_proc=ModelProcessing.ENSEMBLE_CREATION)


@pytest.mark.parametrize("ensemble_inference_on_set", [(True, False, False), (False, True, False), (False, False, True)])
def test_ensemble_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                                 ensemble_inference_on_set: Tuple[bool, bool, bool]) -> None:
    """
    Test inference overrides with ModelProcessing.ENSEMBLE_CREATION.

    :param test_output_dirs: Test output directories.
    :param perform_cross_validation: Whether to test with cross validation.
    :param ensemble_inference_on_set: Overrides for inference on data sets.
    :return: None.
    """
    (ensemble_inference_on_train_set, ensemble_inference_on_val_set, ensemble_inference_on_test_set) = ensemble_inference_on_set
    run_model_inference_train_and_test(test_output_dirs,
                                       True,
                                       ensemble_inference_on_train_set=ensemble_inference_on_train_set,
                                       ensemble_inference_on_val_set=ensemble_inference_on_val_set,
                                       ensemble_inference_on_test_set=ensemble_inference_on_test_set,
                                       model_proc=ModelProcessing.ENSEMBLE_CREATION)


def run_model_inference_train_and_test(test_output_dirs: OutputFolderForTests,
                                       perform_cross_validation: bool,
                                       inference_on_train_set: Optional[bool] = None,
                                       inference_on_val_set: Optional[bool] = None,
                                       inference_on_test_set: Optional[bool] = None,
                                       ensemble_inference_on_train_set: Optional[bool] = None,
                                       ensemble_inference_on_val_set: Optional[bool] = None,
                                       ensemble_inference_on_test_set: Optional[bool] = None,
                                       model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> None:
    """
    Test running inference produces expected output metrics, files, folders and calls to upload_folder.

    :param test_output_dirs: Test output directories.
    :param perform_cross_validation: Whether to test with cross validation.
    :param inference_on_train_set: Override for inference on train data sets.
    :param inference_on_val_set: Override for inference on validation data sets.
    :param inference_on_test_set: Override for inference on test data sets.
    :param ensemble_inference_on_train_set: Override for ensemble inference on train data sets.
    :param ensemble_inference_on_val_set: Override for ensemble inference on validation data sets.
    :param ensemble_inference_on_test_set: Override for ensemble inference on test data sets.
    :param model_proc: Model processing to test.
    :return: None.
    """
    dummy_model = DummyModel()

    config = PassThroughModel()
    # Copy settings from DummyModel
    config.image_channels = dummy_model.image_channels
    config.ground_truth_ids = dummy_model.ground_truth_ids
    config.ground_truth_ids_display_names = dummy_model.ground_truth_ids_display_names
    config.colours = dummy_model.colours
    config.fill_holes = dummy_model.fill_holes
    config.roi_interpreted_types = dummy_model.roi_interpreted_types

    config.test_crop_size = (16, 16, 16)
    config.number_of_cross_validation_splits = 2 if perform_cross_validation else 0
    config.inference_on_train_set = inference_on_train_set
    config.inference_on_val_set = inference_on_val_set
    config.inference_on_test_set = inference_on_test_set
    config.ensemble_inference_on_train_set = ensemble_inference_on_train_set
    config.ensemble_inference_on_val_set = ensemble_inference_on_val_set
    config.ensemble_inference_on_test_set = ensemble_inference_on_test_set
    # Plotting crashes with random TCL errors on Windows, disable that for Windows PR builds.
    config.is_plotting_enabled = common_util.is_linux()

    config.set_output_to(test_output_dirs.root_dir)
    train_and_test_data_small_dir = test_output_dirs.root_dir / "train_and_test_data_small"
    config.local_dataset = create_train_and_test_data_small_dataset(config.test_crop_size,
                                                                    full_ml_test_data_path(),
                                                                    "train_and_test_data",
                                                                    train_and_test_data_small_dir,
                                                                    "data")

    checkpoint_path = config.checkpoint_folder / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    create_model_and_store_checkpoint(config, checkpoint_path)
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    checkpoint_handler.additional_training_done()

    mock_upload_path = test_output_dirs.root_dir / "mock_upload"
    mock_upload_path.mkdir()

    run = create_mock_run(mock_upload_path, config)

    azure_config = Mock(name='mock_azure_config')
    azure_config.fetch_run.return_value = run

    runner = MLRunner(model_config=config, azure_config=azure_config)

    with mock.patch("InnerEye.ML.model_testing.PARENT_RUN_CONTEXT", run):
        metrics = runner.model_inference_train_and_test(
            checkpoint_paths=checkpoint_handler.get_checkpoints_to_test(),
            model_proc=model_proc)

    if model_proc == ModelProcessing.ENSEMBLE_CREATION:
        # Create a fake ensemble dataset.csv
        dataset_df = create_dataset_df()
        dataset_df.to_csv(config.outputs_folder / DATASET_CSV_FILE_NAME)

        with mock.patch.object(PlotCrossValidationConfig, 'azure_config', return_value=azure_config):
            with mock.patch("InnerEye.Azure.azure_util.PARENT_RUN_CONTEXT", run):
                with mock.patch("InnerEye.ML.run_ml.PARENT_RUN_CONTEXT", run):
                    runner.plot_cross_validation_and_upload_results()
                    runner.generate_report(ModelProcessing.ENSEMBLE_CREATION)

    if model_proc == ModelProcessing.DEFAULT:
        named_metrics = {
            ModelExecutionMode.TRAIN: inference_on_train_set,
            ModelExecutionMode.TEST: inference_on_test_set,
            ModelExecutionMode.VAL: inference_on_val_set
        }
    else:
        named_metrics = {
            ModelExecutionMode.TRAIN: ensemble_inference_on_train_set,
            ModelExecutionMode.TEST: ensemble_inference_on_test_set,
            ModelExecutionMode.VAL: ensemble_inference_on_val_set
        }

    error = ''
    expected_upload_folder_count = 0
    for mode, flag in named_metrics.items():
        if mode in metrics:
            metric = metrics[mode]
            assert isinstance(metric, InferenceMetricsForSegmentation)

        if flag is None:
            # No override supplied, calculate the expected default:
            if model_proc == ModelProcessing.DEFAULT:
                if not perform_cross_validation:
                    # If a "normal" run then default to val or test.
                    flag = mode == ModelExecutionMode.TEST
                else:
                    # If an ensemble child then default to never.
                    flag = False
            else:
                # If an ensemble then default to test only.
                flag = mode == ModelExecutionMode.TEST

        if mode in metrics and not flag:
            error = error + f"Error: {mode.value} cannot be not None."
        elif mode not in metrics and flag:
            error = error + f"Error: {mode.value} cannot be None."
        results_folder = config.outputs_folder / get_best_epoch_results_path(mode, model_proc)
        folder_exists = results_folder.is_dir()
        assert folder_exists == flag
        if flag and model_proc == ModelProcessing.ENSEMBLE_CREATION:
            expected_upload_folder_count = expected_upload_folder_count + 1
            expected_name = get_best_epoch_results_path(mode, ModelProcessing.DEFAULT)
            run.upload_folder.assert_any_call(name=str(expected_name), path=str(results_folder))
    if len(error):
        raise ValueError(error)

    if model_proc == ModelProcessing.ENSEMBLE_CREATION:
        # The report should have been mock uploaded
        expected_upload_folder_count = expected_upload_folder_count + 1

    assert run.upload_folder.call_count == expected_upload_folder_count


def test_logging_to_file(test_output_dirs: OutputFolderForTests) -> None:
    # Log file should go to a new, non-existent folder, 2 levels deep
    file_path = test_output_dirs.root_dir / "subdir1" / "subdir2" / "logfile.txt"
    common_util.logging_to_file_handler = None
    common_util.logging_to_file(file_path)
    assert common_util.logging_to_file_handler is not None
    log_line = "foo bar"
    logging.getLogger().setLevel(logging.INFO)
    logging.info(log_line)
    common_util.disable_logging_to_file()
    should_not_be_present = "This should not be present in logs"
    logging.info(should_not_be_present)
    assert common_util.logging_to_file_handler is None
    # Wait for a bit, tests sometimes fail with the file not existing yet
    time.sleep(2)
    assert file_path.exists()
    assert log_line in file_path.read_text()
    assert should_not_be_present not in file_path.read_text()


def test_cross_validation_for_lighting_container_models_is_supported() -> None:
    """
    Prior to https://github.com/microsoft/InnerEye-DeepLearning/pull/483 we raised an exception in
    runner.run when cross validation was attempted on a lightning container. This test checks that
    we do not raise the exception anymore, and instead pass on a cross validation hyperdrive config
    to azure_runner's submit_to_azureml method.
    """
    args_list = ["--model=HelloContainer", "--number_of_cross_validation_splits=5", "--azureml=True"]
    mock_run = mock.MagicMock()
    mock_run.id = "foo"
    mock_run.experiment.name = "bar"
    with mock.patch("sys.argv", [""] + args_list):
        runner = Runner(project_root=fixed_paths.repository_root_directory(),
                        yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
        with mock.patch("health_azure.himl.submit_run", return_value=mock_run) as create_and_submit_experiment_patch:
            with pytest.raises(SystemExit):
                runner.run()
            assert runner.lightning_container.model_name == 'HelloContainer'
            assert runner.lightning_container.number_of_cross_validation_splits == 5
            script_run_config_arg = create_and_submit_experiment_patch.call_args[1]["script_run_config"]
            assert isinstance(script_run_config_arg, HyperDriveConfig)

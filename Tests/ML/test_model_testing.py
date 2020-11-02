#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from InnerEye.Common import common_util
from InnerEye.Common.common_util import get_epoch_results_path
from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML import model_testing
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import DATASET_ID_FILE, GROUND_TRUTH_IDS_FILE
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME, create_inference_pipeline
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import InferencePipeline
from InnerEye.ML.pipelines.scalar_inference import ScalarEnsemblePipeline, ScalarInferencePipeline
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.visualizers.plot_cross_validation import get_config_and_results_for_offline_runs
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import assert_file_contains_string, assert_text_files_match, assert_nifti_content, \
    get_image_shape
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
def test_model_test(test_output_dirs: TestOutputDirectories) -> None:
    train_and_test_data_dir = full_ml_test_data_path("train_and_test_data")

    config = DummyModel()
    config.set_output_to(test_output_dirs.root_dir)
    epoch = 1
    config.num_epochs = epoch
    assert config.get_test_epochs() == [epoch]
    placeholder_dataset_id = "place_holder_dataset_id"
    config.azure_dataset_id = placeholder_dataset_id
    # Mimic the behaviour that checkpoints are downloaded from blob storage into the checkpoints folder.
    stored_checkpoints = full_ml_test_data_path("checkpoints")
    shutil.copytree(str(stored_checkpoints), str(config.checkpoint_folder))
    transform = config.get_full_image_sample_transforms().test
    df = pd.read_csv(full_ml_test_data_path(DATASET_CSV_FILE_NAME))
    df = df[df.subject.isin([1, 2])]
    # noinspection PyTypeHints
    config._datasets_for_inference = \
        {ModelExecutionMode.TEST: FullImageDataset(config, df, full_image_sample_transforms=transform)}  # type: ignore
    execution_mode = ModelExecutionMode.TEST
    inference_results = model_testing.segmentation_model_test(config, execution_mode)
    epoch_dir = config.outputs_folder / get_epoch_results_path(epoch, execution_mode)
    assert inference_results.epochs[epoch] == pytest.approx(0.66606902, abs=1e-6)

    assert config.outputs_folder.is_dir()
    assert epoch_dir.is_dir()
    patient1 = io_util.load_nifti_image(train_and_test_data_dir / "id1_channel1.nii.gz")
    patient2 = io_util.load_nifti_image(train_and_test_data_dir / "id2_channel1.nii.gz")

    assert_file_contains_string(epoch_dir / DATASET_ID_FILE, placeholder_dataset_id)
    assert_file_contains_string(epoch_dir / GROUND_TRUTH_IDS_FILE, "region")
    assert_text_files_match(epoch_dir / model_testing.METRICS_FILE_NAME,
                            Path(train_and_test_data_dir) / model_testing.METRICS_FILE_NAME)
    assert_text_files_match(epoch_dir / model_testing.METRICS_AGGREGATES_FILE,
                            Path(train_and_test_data_dir) / model_testing.METRICS_AGGREGATES_FILE)
    # Plotting results vary between platforms. Can only check if the file is generated, but not its contents.
    assert (epoch_dir / model_testing.BOXPLOT_FILE).exists()

    assert_nifti_content(epoch_dir / "001" / "posterior_region.nii.gz", get_image_shape(patient1),
                         patient1.header,
                         [136], np.ubyte)
    assert_nifti_content(epoch_dir / "002" / "posterior_region.nii.gz", get_image_shape(patient2),
                         patient2.header,
                         [136], np.ubyte)
    assert_nifti_content(epoch_dir / "001" / DEFAULT_RESULT_IMAGE_NAME, get_image_shape(patient1),
                         patient1.header,
                         [1], np.ubyte)
    assert_nifti_content(epoch_dir / "002" / DEFAULT_RESULT_IMAGE_NAME, get_image_shape(patient2),
                         patient2.header,
                         [1], np.ubyte)
    assert_nifti_content(epoch_dir / "001" / "posterior_background.nii.gz", get_image_shape(patient1),
                         patient1.header,
                         [118], np.ubyte)
    assert_nifti_content(epoch_dir / "002" / "posterior_background.nii.gz", get_image_shape(patient2),
                         patient2.header,
                         [118], np.ubyte)
    thumbnails_folder = epoch_dir / model_testing.THUMBNAILS_FOLDER
    assert thumbnails_folder.is_dir()
    png_files = list(thumbnails_folder.glob("*.png"))
    overlays = [f for f in png_files if "_region_slice_" in str(f)]
    assert len(overlays) == len(df.subject.unique()), "There should be one overlay/contour file per subject"

    # Writing dataset.csv normally happens at the beginning of training,
    # but this test reads off a saved checkpoint file.
    # Dataset.csv must be present for plot_cross_validation.
    config.write_dataset_files()
    # Test if the metrics files can be picked up correctly by the cross validation code
    config_and_files = get_config_and_results_for_offline_runs(config)
    result_files = config_and_files.files
    assert len(result_files) == 1
    for file in result_files:
        assert file.execution_mode == execution_mode
        assert file.dataset_csv_file is not None
        assert file.dataset_csv_file.exists()
        assert file.metrics_file is not None
        assert file.metrics_file.exists()


@pytest.mark.parametrize(("config", "checkpoint_folder"),
                         [(DummyModel(), "checkpoints"),
                          (ClassificationModelForTesting(), "classification_data_generated_random/checkpoints")])
def test_create_inference_pipeline_invalid_epoch(config: ModelConfigBase,
                                                 checkpoint_folder: str,
                                                 test_output_dirs: TestOutputDirectories) -> None:
    config.set_output_to(test_output_dirs.root_dir)
    # Mimic the behaviour that checkpoints are downloaded from blob storage into the checkpoints folder.
    stored_checkpoints = full_ml_test_data_path(checkpoint_folder)
    shutil.copytree(str(stored_checkpoints), str(config.checkpoint_folder))
    # no pipeline created when checkpoint for epoch does not exist
    assert create_inference_pipeline(config, 10) is None


@pytest.mark.parametrize("with_run_recovery", [False, True])
@pytest.mark.parametrize(("config", "checkpoint_folder", "inference_type", "ensemble_type"),
                         [(DummyModel(), "checkpoints", InferencePipeline, EnsemblePipeline),
                          (ClassificationModelForTesting(mean_teacher_model=False),
                           "classification_data_generated_random/checkpoints",
                           ScalarInferencePipeline, ScalarEnsemblePipeline),
                          (ClassificationModelForTesting(mean_teacher_model=True),
                           "classification_data_generated_random/checkpoints",
                           ScalarInferencePipeline, ScalarEnsemblePipeline)
                          ])
def test_create_inference_pipeline(with_run_recovery: bool,
                                   config: ModelConfigBase,
                                   checkpoint_folder: str,
                                   inference_type: type,
                                   ensemble_type: type,
                                   test_output_dirs: TestOutputDirectories) -> None:
    config.set_output_to(test_output_dirs.root_dir)
    # Mimic the behaviour that checkpoints are downloaded from blob storage into the checkpoints folder.
    stored_checkpoints = full_ml_test_data_path(checkpoint_folder)
    shutil.copytree(str(stored_checkpoints), str(config.checkpoint_folder))

    if with_run_recovery:
        run_recovery: Optional[RunRecovery] = RunRecovery(checkpoints_roots=[stored_checkpoints])
    else:
        run_recovery = None
    assert isinstance(create_inference_pipeline(config, 1, run_recovery), inference_type)

    # test for ensemble pipeline if run_recovery is enabled
    if with_run_recovery:
        run_recovery = RunRecovery(checkpoints_roots=[stored_checkpoints] * 2)
        assert isinstance(create_inference_pipeline(config, 1, run_recovery), ensemble_type)

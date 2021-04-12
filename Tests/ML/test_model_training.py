#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import is_windows, logging_to_stdout
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.metrics_constants import MetricType, TrackedMetrics, VALIDATION_PREFIX
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, DATASET_CSV_FILE_NAME, ModelExecutionMode, \
    RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX, \
    STORED_CSV_FILE_NAMES
from InnerEye.ML.config import MixtureLossComponent, SegmentationLoss
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.lightning_loggers import StoringLogger
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.utils.io_util import load_nifti_image
from InnerEye.ML.utils.model_util import create_segmentation_loss_function
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.visualizers.patch_sampling import PATCH_SAMPLING_FOLDER
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import machine_has_gpu, model_train_unittest, get_default_checkpoint_handler

config_path = full_ml_test_data_path()
base_path = full_ml_test_data_path()


def test_get_total_number_of_training_epochs() -> None:
    c = DeepLearningConfig(num_epochs=2, should_validate=False)
    assert c.get_total_number_of_training_epochs() == 2
    c = DeepLearningConfig(num_epochs=10, start_epoch=5, should_validate=False)
    assert c.get_total_number_of_training_epochs() == 5


@pytest.mark.parametrize("image_channels", [["region"], ["random_123"]])
@pytest.mark.parametrize("ground_truth_ids", [["region", "region"], ["region", "other_region"]])
def test_invalid_model_train(test_output_dirs: OutputFolderForTests, image_channels: Any,
                             ground_truth_ids: Any) -> None:
    with pytest.raises(ValueError):
        _test_model_train(test_output_dirs, image_channels, ground_truth_ids)


@pytest.mark.cpu_and_gpu
@pytest.mark.parametrize("no_mask_channel", [True, False])
def test_valid_model_train(test_output_dirs: OutputFolderForTests, no_mask_channel: bool) -> None:
    _test_model_train(test_output_dirs, ["channel1", "channel2"], ["region", "region_1"], no_mask_channel)


def _test_model_train(output_dirs: OutputFolderForTests,
                      image_channels: Any,
                      ground_truth_ids: Any,
                      no_mask_channel: bool = False) -> None:
    def _check_patch_centers(diagnostics_per_epoch: List[np.ndarray], should_equal: bool) -> None:
        patch_centers_epoch1 = diagnostics_per_epoch[0]
        assert len(diagnostics_per_epoch) > 1, "Not enough data to check patch centers, need at least 2"
        for diagnostic in diagnostics_per_epoch[1:]:
            assert np.array_equal(patch_centers_epoch1, diagnostic) == should_equal

    def _check_voxel_count(results_per_epoch: List[Dict[str, float]],
                           expected_voxel_count_per_epoch: List[float],
                           prefix: str) -> None:
        assert len(results_per_epoch) == len(expected_voxel_count_per_epoch)
        for epoch, (results, voxel_count) in enumerate(zip(results_per_epoch, expected_voxel_count_per_epoch)):
            # In the test data, both structures "region" and "region_1" are read from the same nifti file, hence
            # their voxel counts must be identical.
            for structure in ["region", "region_1"]:
                assert results[f"{MetricType.VOXEL_COUNT.value}/{structure}"] == pytest.approx(voxel_count, abs=1e-2), \
                    f"{prefix} voxel count mismatch for '{structure}' epoch {epoch}"

    def _mean(a: List[float]) -> float:
        return sum(a) / len(a)

    def _mean_list(lists: List[List[float]]) -> List[float]:
        return list(map(_mean, lists))

    logging_to_stdout(log_level=logging.DEBUG)
    train_config = DummyModel()
    train_config.local_dataset = base_path
    train_config.set_output_to(output_dirs.root_dir)
    train_config.image_channels = image_channels
    train_config.ground_truth_ids = ground_truth_ids
    train_config.mask_id = None if no_mask_channel else train_config.mask_id
    train_config.random_seed = 42
    train_config.class_weights = [0.5, 0.25, 0.25]
    train_config.store_dataset_sample = True
    train_config.recovery_checkpoint_save_interval = 1

    if machine_has_gpu:
        expected_train_losses = [0.4553468, 0.454904]
        expected_val_losses = [0.4553881, 0.4553041]
    else:
        expected_train_losses = [0.4553469, 0.4548947]
        expected_val_losses = [0.4553880, 0.4553041]
    loss_absolute_tolerance = 1e-6
    expected_learning_rates = [train_config.l_rate, 5.3589e-4]

    model_training_result, _ = model_train_unittest(train_config, dirs=output_dirs)
    assert isinstance(model_training_result, StoringLogger)

    def assert_all_close(metric: str, expected: List[float], **kwargs: Any) -> None:
        actual = model_training_result.get_train_metric(metric)
        assert np.allclose(actual, expected, **kwargs), f"Mismatch for {metric}: Got {actual}, expected {expected}"

    # check to make sure training batches are NOT all the same across epochs
    _check_patch_centers(model_training_result.train_diagnostics, should_equal=False)
    # check to make sure validation batches are all the same across epochs
    _check_patch_centers(model_training_result.val_diagnostics, should_equal=True)
    assert_all_close(MetricType.SUBJECT_COUNT.value, [3.0, 3.0])
    assert_all_close(MetricType.LEARNING_RATE.value, expected_learning_rates, rtol=1e-6)

    if is_windows():
        # Randomization comes out slightly different on Windows. Skip the rest of the detailed checks.
        return

    # Simple regression test: Voxel counts should be the same in both epochs on the validation set,
    # and be the same across 'region' and 'region_1' because they derive from the same Nifti files.
    # The following values are read off directly from the results of compute_dice_across_patches in the training loop
    # This checks that averages are computed correctly, and that metric computers are reset after each epoch.
    train_voxels = [[83092.0, 83212.0, 82946.0], [83000.0, 82881.0, 83309.0]]
    val_voxels = [[82765.0, 83212.0], [82765.0, 83212.0]]
    _check_voxel_count(model_training_result.train_results_per_epoch(), _mean_list(train_voxels), "Train")
    _check_voxel_count(model_training_result.val_results_per_epoch(), _mean_list(val_voxels), "Val")

    actual_train_losses = model_training_result.get_train_metric(MetricType.LOSS.value)
    actual_val_losses = model_training_result.get_val_metric(MetricType.LOSS.value)
    print("actual_train_losses = {}".format(actual_train_losses))
    print("actual_val_losses = {}".format(actual_val_losses))
    assert np.allclose(actual_train_losses, expected_train_losses, atol=loss_absolute_tolerance), "Train losses"
    assert np.allclose(actual_val_losses, expected_val_losses, atol=loss_absolute_tolerance), "Val losses"
    # Check that the metric we track for Hyperdrive runs is actually written.
    assert TrackedMetrics.Val_Loss.value.startswith(VALIDATION_PREFIX)
    tracked_metric = TrackedMetrics.Val_Loss.value[len(VALIDATION_PREFIX):]
    for val_result in model_training_result.val_results_per_epoch():
        assert tracked_metric in val_result

    # The following values are read off directly from the results of compute_dice_across_patches in the
    # training loop. Results are slightly different for CPU, hence use a larger tolerance there.
    dice_tolerance = 1e-4 if machine_has_gpu else 4.5e-4
    train_dice_region = [[0.0, 0.0, 4.0282e-04], [0.0309, 0.0334, 0.0961]]
    train_dice_region1 = [[0.4806, 0.4800, 0.4832], [0.4812, 0.4842, 0.4663]]
    # There appears to be some amount of non-determinism here: When using a tolerance of 1e-4, we get occasional
    # test failures on Linux in the cloud (not on Windows, not on AzureML) Unclear where it comes from. Even when
    # failing here, the losses match up to the expected tolerance.
    assert_all_close("Dice/region", _mean_list(train_dice_region), atol=dice_tolerance)
    assert_all_close("Dice/region_1", _mean_list(train_dice_region1), atol=dice_tolerance)
    expected_average_dice = [_mean(train_dice_region[i] + train_dice_region1[i])  # type: ignore
                             for i in range(len(train_dice_region))]
    assert_all_close("Dice/AverageAcrossStructures", expected_average_dice, atol=dice_tolerance)

    # check output files/directories
    assert train_config.outputs_folder.is_dir()
    assert train_config.logs_folder.is_dir()

    # Tensorboard event files go into a Lightning subfolder (Pytorch Lightning default)
    assert (train_config.logs_folder / "Lightning").is_dir()
    assert len([(train_config.logs_folder / "Lightning").glob("events*")]) == 1

    assert train_config.num_epochs == 2
    # Checkpoint folder
    assert train_config.checkpoint_folder.is_dir()
    actual_checkpoints = list(train_config.checkpoint_folder.rglob("*.ckpt"))
    assert len(actual_checkpoints) == 2, f"Actual checkpoints: {actual_checkpoints}"
    assert (train_config.checkpoint_folder / RECOVERY_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()
    assert (train_config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX).is_file()
    assert (train_config.outputs_folder / DATASET_CSV_FILE_NAME).is_file()
    assert (train_config.outputs_folder / STORED_CSV_FILE_NAMES[ModelExecutionMode.TRAIN]).is_file()
    assert (train_config.outputs_folder / STORED_CSV_FILE_NAMES[ModelExecutionMode.VAL]).is_file()

    # Path visualization: There should be 3 slices for each of the 2 subjects
    sampling_folder = train_config.outputs_folder / PATCH_SAMPLING_FOLDER
    assert sampling_folder.is_dir()
    assert train_config.show_patch_sampling > 0
    assert len(list(sampling_folder.rglob("*.png"))) == 3 * train_config.show_patch_sampling

    # Time per epoch: Test that we have all these times logged.
    model_training_result.get_train_metric(MetricType.SECONDS_PER_EPOCH.value)
    model_training_result.get_val_metric(MetricType.SECONDS_PER_EPOCH.value)
    model_training_result.get_val_metric(MetricType.SECONDS_PER_BATCH.value)
    model_training_result.get_train_metric(MetricType.SECONDS_PER_BATCH.value)

    # Issue #372
    # # Test for saving of example images
    # assert train_config.example_images_folder.is_dir()
    # example_files = list(train_config.example_images_folder.rglob("*.*"))
    # assert len(example_files) == 3 * 2


def test_create_data_loaders() -> None:
    train_config = DummyModel()
    create_data_loaders(train_config)


def create_data_loaders(train_config: DummyModel) -> None:
    train_config.train_batch_size = 1
    train_config.local_dataset = base_path
    # create the dataset splits
    dataset_splits = train_config.get_dataset_splits()
    # create the data loaders
    data_loaders = train_config.create_data_loaders()
    train_loader = data_loaders[ModelExecutionMode.TRAIN]
    val_loader = data_loaders[ModelExecutionMode.VAL]

    def check_patient_id_in_dataset(loader: DataLoader, split: pd.DataFrame) -> None:
        subjects = list(split.subject.unique())
        for i, x in enumerate(loader):
            sample_from_loader = CroppedSample.from_dict(x)
            assert isinstance(sample_from_loader.metadata, list)
            assert len(sample_from_loader.metadata) == 1
            assert sample_from_loader.metadata[0].patient_id in subjects

    # check if the subjects in the dataloaders are the same in the corresponding dataset splits
    for loader, split in [(train_loader, dataset_splits.train), (val_loader, dataset_splits.val)]:
        check_patient_id_in_dataset(loader, split)


def test_create_data_loaders_hdf5(test_output_dirs: OutputFolderForTests) -> None:
    dataset_dir = convert_nifti_data_to_hdf5(test_output_dirs.root_dir)
    train_config = DummyModel()
    train_config.local_dataset = dataset_dir
    create_data_loaders(train_config)


def convert_nifti_data_to_hdf5(output_hdf5_dir: Path) -> Path:
    # create dataset in hdf5
    csv_str = (base_path / "dataset.csv").read_text()
    csv_str = csv_str.replace("train_and_test_data/id1_channel1.nii.gz,channel1",
                              "p1.h5|volume|0,channel1")
    csv_str = csv_str.replace("train_and_test_data/id1_channel1.nii.gz,channel2",
                              "p1.h5|volume|1,channel2")
    csv_str = csv_str.replace("train_and_test_data/id2_channel1.nii.gz,channel1",
                              "p2.h5|volume|0,channel1")
    csv_str = csv_str.replace("train_and_test_data/id2_channel1.nii.gz,channel2",
                              "p2.h5|volume|1,channel2")
    # segmentation
    csv_str = csv_str.replace("train_and_test_data/id1_region.nii.gz,region",
                              "p1.h5|region|0,region")
    csv_str = csv_str.replace("train_and_test_data/id1_region.nii.gz,region_1",
                              "p2.h5|region|0,region_1")
    csv_str = csv_str.replace("train_and_test_data/id2_region.nii.gz,region",
                              "p2.h5|region|0,region")
    csv_str = csv_str.replace("train_and_test_data/id2_region.nii.gz,region_1",
                              "p2.h5|region_1|1,region_1")
    # mask
    csv_str = csv_str.replace("train_and_test_data/id1_mask.nii.gz,mask",
                              "p1.h5|mask|0,mask")
    csv_str = csv_str.replace("train_and_test_data/id2_mask.nii.gz,mask",
                              "p2.h5|mask|0,mask")

    dataset_dir = output_hdf5_dir / "hdf5_dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset.csv").write_text(csv_str)
    train_data = base_path / "train_and_test_data"
    create_hdf5_from_nifti(train_data / "id1_channel1.nii.gz", train_data / "id1_region.nii.gz",
                           train_data / "id1_mask.nii.gz",
                           dataset_dir / "p1.h5")
    create_hdf5_from_nifti(train_data / "id2_channel1.nii.gz", train_data / "id2_region.nii.gz",
                           train_data / "id2_mask.nii.gz",
                           dataset_dir / "p2.h5")
    return dataset_dir


def create_hdf5_from_nifti(input_nifti_volume: Path, input_nifti_seg: Path, input_nifti_mask: Path,
                           output_h5: Path) -> None:
    volume = load_nifti_image(input_nifti_volume).image
    volume_with_channels = np.expand_dims(volume, axis=0)
    volume_with_channels = np.resize(volume_with_channels, (2,) + volume_with_channels.shape[1:])
    segmentation = load_nifti_image(input_nifti_seg).image
    seg_with_channels = np.expand_dims(segmentation, axis=0)
    mask = load_nifti_image(input_nifti_mask).image
    mask_with_channels = np.expand_dims(mask, axis=0)
    with h5py.File(str(output_h5), 'w') as hf:
        hf.create_dataset('volume', data=volume_with_channels, compression="gzip", compression_opts=9)
        hf.create_dataset('region', data=seg_with_channels, compression="gzip", compression_opts=9)
        hf.create_dataset('region_1', data=seg_with_channels, compression="gzip", compression_opts=9)
        hf.create_dataset('mask', data=mask_with_channels, compression="gzip", compression_opts=9)


def test_construct_loss_function() -> None:
    model_config = DummyModel()
    model_config.loss_type = SegmentationLoss.Mixture
    # Weights deliberately do not sum to 1.0.
    weights = [1.5, 0.5]
    model_config.mixture_loss_components = [
        MixtureLossComponent(weights[0], SegmentationLoss.CrossEntropy, 0.2),
        MixtureLossComponent(weights[1], SegmentationLoss.SoftDice, 0.1)]
    loss_fn = create_segmentation_loss_function(model_config)
    assert isinstance(loss_fn, MixtureLoss)
    assert len(loss_fn.components) == len(weights)
    assert loss_fn.components[0][0] == weights[0] / sum(weights)
    assert loss_fn.components[1][0] == weights[1] / sum(weights)


def test_recover_training_mean_teacher_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests that training can be recovered from a previous checkpoint.
    """
    config = DummyClassification()
    config.mean_teacher_alpha = 0.999
    config.recovery_checkpoint_save_interval = 1
    config.set_output_to(test_output_dirs.root_dir / "original")
    os.makedirs(str(config.outputs_folder))

    original_checkpoint_folder = config.checkpoint_folder

    # First round of training
    config.num_epochs = 2
    model_train_unittest(config, dirs=test_output_dirs)
    assert len(list(config.checkpoint_folder.glob("*.*"))) == 2

    # Restart training from previous run
    config.start_epoch = 2
    config.num_epochs = 3
    config.set_output_to(test_output_dirs.root_dir / "recovered")
    os.makedirs(str(config.outputs_folder))
    # make if seem like run recovery objects have been downloaded
    checkpoint_root = config.checkpoint_folder / "old_run"
    shutil.copytree(str(original_checkpoint_folder), str(checkpoint_root))

    # Create a new checkpoint handler and set run_recovery to the copied checkpoints
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    checkpoint_handler.run_recovery = RunRecovery([checkpoint_root])

    model_train_unittest(config, dirs=test_output_dirs, checkpoint_handler=checkpoint_handler)
    # remove recovery checkpoints
    shutil.rmtree(checkpoint_root)
    assert len(list(config.checkpoint_folder.glob("*.*"))) == 2


def test_script_names_correct() -> None:
    for file in [*fixed_paths.SCRIPTS_AT_ROOT, fixed_paths.RUN_SCORING_SCRIPT]:
        full_file = fixed_paths.repository_root_directory() / file
        assert full_file.exists(), f"{file} does not exist."

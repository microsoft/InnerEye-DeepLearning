#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from pathlib import Path
from typing import Any, List

import h5py
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from InnerEye.Common import fixed_paths
from InnerEye.Common.metrics_dict import MetricType, MetricsDict
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import metrics, model_training
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode, STORED_CSV_FILE_NAMES
from InnerEye.ML.config import MixtureLossComponent, SegmentationLoss
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.dataset.sample import CroppedSample
from InnerEye.ML.deep_learning_config import DeepLearningConfig, TemperatureScalingConfig
from InnerEye.ML.model_training import model_train
from InnerEye.ML.models.losses.mixture import MixtureLoss
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.io_util import load_nifti_image
from InnerEye.ML.utils.model_util import create_segmentation_loss_function
from InnerEye.ML.utils.run_recovery import RunRecovery
from InnerEye.ML.utils.training_util import ModelTrainingResults
from InnerEye.ML.visualizers.patch_sampling import PATCH_SAMPLING_FOLDER
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import assert_file_contains_string, get_default_checkpoint_handler
from Tests.fixed_paths_for_tests import full_ml_test_data_path

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

    train_config = DummyModel()
    train_config.local_dataset = base_path
    train_config.set_output_to(output_dirs.root_dir)
    train_config.image_channels = image_channels
    train_config.ground_truth_ids = ground_truth_ids
    train_config.mask_id = None if no_mask_channel else train_config.mask_id
    train_config.random_seed = 42
    train_config.class_weights = [0.5, 0.25, 0.25]
    train_config.store_dataset_sample = True
    train_config.save_step_epochs = 2

    expected_train_losses = [0.455572, 0.455031]
    expected_val_losses = [0.455479, 0.455430]
    loss_absolute_tolerance = 1e-6
    expected_learning_rates = [train_config.l_rate, 5.3589e-4]

    checkpoint_handler = get_default_checkpoint_handler(model_config=train_config,
                                                        project_root=Path(output_dirs.root_dir))
    model_training_result = model_training.model_train(train_config,
                                                       checkpoint_handler=checkpoint_handler)
    assert isinstance(model_training_result, ModelTrainingResults)

    # check to make sure training batches are NOT all the same across epochs
    _check_patch_centers(model_training_result.train_diagnostics, should_equal=False)
    # check to make sure validation batches are all the same across epochs
    _check_patch_centers(model_training_result.val_diagnostics, should_equal=True)
    actual_train_losses = model_training_result.get_metric(is_training=True, metric_type=MetricType.LOSS)
    actual_val_losses = model_training_result.get_metric(is_training=False, metric_type=MetricType.LOSS)
    actual_learning_rates = model_training_result.get_metric(is_training=True, metric_type=MetricType.LEARNING_RATE)
    print("actual_train_losses = {}".format(actual_train_losses))
    print("actual_val_losses = {}".format(actual_val_losses))
    assert np.allclose(actual_train_losses, expected_train_losses, atol=loss_absolute_tolerance), "Train losses"
    assert np.allclose(actual_val_losses, expected_val_losses, atol=loss_absolute_tolerance), "Val losses"
    assert np.allclose(actual_learning_rates, expected_learning_rates, rtol=1e-6), "Learning rates"

    # check output files/directories
    assert train_config.outputs_folder.is_dir()
    assert train_config.logs_folder.is_dir()

    # Tensorboard event files go into a Lightning subfolder (Pytorch Lightning default)
    assert (train_config.logs_folder / "Lightning").is_dir()
    assert len([(train_config.logs_folder / "Lightning").glob("events*")]) == 1

    assert train_config.num_epochs == 2
    # Checkpoint folder
    assert train_config.checkpoint_folder.is_dir()
    actual_checkpoints =list(train_config.checkpoint_folder.rglob("*.ckpt"))
    assert len(actual_checkpoints) == 2, f"Actual checkpoints: {actual_checkpoints}"
    assert (train_config.checkpoint_folder / "epoch=1.ckpt").is_file()
    assert (train_config.checkpoint_folder / "best_val_loss-v0.ckpt").is_file()
    assert (train_config.outputs_folder / DATASET_CSV_FILE_NAME).is_file()
    assert (train_config.outputs_folder / STORED_CSV_FILE_NAMES[ModelExecutionMode.TRAIN]).is_file()
    assert (train_config.outputs_folder / STORED_CSV_FILE_NAMES[ModelExecutionMode.VAL]).is_file()

    # Path visualization: There should be 3 slices for each of the 2 subjects
    sampling_folder = train_config.outputs_folder / PATCH_SAMPLING_FOLDER
    assert sampling_folder.is_dir()
    assert train_config.show_patch_sampling > 0
    assert len(list(sampling_folder.rglob("*.png"))) == 3 * train_config.show_patch_sampling

    # TODO antonsc: enable
    # # Test for saving of example images
    # assert train_config.example_images_folder.is_dir()
    # example_files = list(train_config.example_images_folder.rglob("*.*"))
    # assert len(example_files) == 3 * 2



@pytest.mark.parametrize(["rates", "expected"],
                         [(None, ""),
                          ([], ""),
                          ([0.0], "0.00e+00"),
                          ([0.000056789], "5.68e-05"),
                          ([0.000536], "5.36e-04"),
                          ([1.23456], "1.23e+00")])
def test_format_learning_rate(rates: Any, expected: str) -> None:
    assert metrics.format_learning_rates(rates) == expected


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

    # First round of training
    config.num_epochs = 2
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    model_train(config, checkpoint_handler=checkpoint_handler)
    assert len(list(config.checkpoint_folder.rglob("*.*"))) == 1

    # Restart training from previous run
    config.start_epoch = 2
    config.num_epochs = 3
    # make if seem like run recovery objects have been downloaded
    checkpoint_root = config.checkpoint_folder / "recovered"
    shutil.copytree(config.checkpoint_folder, checkpoint_root)
    checkpoint_handler.run_recovery = RunRecovery([checkpoint_root])

    model_train(config, checkpoint_handler=checkpoint_handler)
    # remove recovery checkpoints
    shutil.rmtree(checkpoint_root)
    assert len(list(config.checkpoint_folder.rglob("*.*"))) == 2


def test_script_names_correct() -> None:
    for file in [*fixed_paths.SCRIPTS_AT_ROOT, fixed_paths.RUN_SCORING_SCRIPT]:
        full_file = fixed_paths.repository_root_directory() / file
        assert full_file.exists(), f"{file} does not exist."

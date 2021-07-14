#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List
from unittest import mock

import numpy as np
from numpy.lib.function_base import append
from sklearn.model_selection import KFold
import torch

from InnerEye.Common.fixed_paths_for_tests import tests_root_directory
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.other.DummyEnsembleRegressionContainer import (DummyEnsembleRegressionContainer,
                                                                        DummyEnsembleRegressionModule)
from InnerEye.ML.configs.other.HelloContainer import HelloContainer, HelloDataModule
from Tests.ML.util import default_runner


# def test_random_ensemble(test_output_dirs: OutputFolderForTests) -> None:
#     """
#     Make dummy checkpoints and load them, test ensemble creates correct files.

#     N.B. we cannot use create_model_and_store_checkpoint as its call to create_lightning_model fails with an
#     AttributeError: 'DummyEnsembleRegressionContainer' object has no attribute 'is_segmentation_model'
#     """
#     # Create checkpoints from initial random weights (i.e. without training)
#     cross_validation_children: List[DummyEnsembleRegressionModule] = []
#     checkpoint_paths: List[Path] = []
#     for idx in range(5):
#         checkpoint_path = test_output_dirs.root_dir / f"{idx}.ckpt"
#         module = DummyEnsembleRegressionModule(outputs_folder=test_output_dirs.root_dir)
#         torch.save({'state_dict': module.model.state_dict()}, checkpoint_path)
#         cross_validation_children.append(module)
#         checkpoint_paths.append(checkpoint_path)
#     # Load checkpoints as ensemble
#     eldest = cross_validation_children[0]
#     eldest.load_checkpoints_as_siblings(checkpoint_paths, use_gpu=False)
#     # Get test data split
#     data_module_xval = HelloDataModule(
#         root_folder=HelloContainer().local_dataset,  # type: ignore
#         cross_validation_split_index=0,
#         number_of_cross_validation_splits=5)
#     test_dataloader = data_module_xval.test_dataloader()
#     # Run inference loop
#     eldest.on_inference_start()
#     eldest.on_inference_start_dataset(execution_mode=ModelExecutionMode.TEST, _=True)
#     for batch_idx, batch in enumerate(test_dataloader):
#         posteriors = eldest.forward(batch['x'])
#         eldest.record_posteriors(batch, batch_idx, posteriors)
#     eldest.on_inference_end_dataset()
#     xval_metrics_dir = test_output_dirs.root_dir / str(ModelExecutionMode.TEST)
#     assert (xval_metrics_dir / "test_mse.txt").exists
#     assert (xval_metrics_dir / "test_mae.txt").exists

def test_trained_ensemble(test_output_dirs: OutputFolderForTests) -> None:
    """
    Make real checkpoints and load them, test ensemble gets better accuracy than any of the child cross validation
    models.
    """
    local_dataset = test_output_dirs.root_dir / "dataset"
    local_dataset.mkdir()
    # Since cross validation for Lightning models will not run locally in our infrastructure we need to set up the data
    # manually for each of the cross validation runs.
    raw_data = np.loadtxt(
        tests_root_directory().parent / "InnerEye" / "ML" / "configs" / "other" / "hellocontainer.csv",
        delimiter=",")
    np.random.seed(42)
    np.random.shuffle(raw_data)
    test_data = raw_data[70:100]
    raw_data_remaining = raw_data[0:70]
    k_fold = KFold(n_splits=5)
    checkpoint_paths: List[Path] = []
    test_mses: List[float] = []
    test_maes: List[float] = []
    first_child: DummyEnsembleRegressionModule = None
    for cross_validation_split_index in range(5):
        runner = default_runner()
        local_dataset = test_output_dirs.root_dir / "dataset" / str(cross_validation_split_index)
        local_dataset.mkdir()
        args = ["", "--model=DummyEnsembleRegressionContainer", "--model_configs_namespace=Tests.ML.configs",
            f"--output_to={test_output_dirs.root_dir}", f"--local_dataset={local_dataset}",
            "--number_of_cross_validation_splits=0"]
        train_indexes, val_indexes = list(k_fold.split(raw_data_remaining))[cross_validation_split_index]
        train_data = raw_data_remaining[train_indexes]
        val_data = raw_data_remaining[val_indexes]
        fold_data = np.concatenate((train_data, val_data, test_data), axis=0)
        np.savetxt(local_dataset / "hellocontainer.csv", fold_data, delimiter=",")
        with mock.patch("sys.argv", args):
            loaded_config, _ = runner.run()
        if not first_child:
            first_child = loaded_config.created_model
            first_child.siblings = [first_child]
        else:
            first_child.siblings.append(loaded_config.created_model)
        checkpoint_path = Path(loaded_config.output_to) / "checkpoints" / "best_checkpoint.ckpt"
        checkpoint_paths.append(checkpoint_path)
        mse_metrics = _load_metrics(metrics_file=loaded_config.file_system_config.run_folder / "test_mse.txt")
        test_mses.append(mse_metrics["TEST"])
        mae_metrics = _load_metrics(metrics_file=loaded_config.file_system_config.run_folder / "test_mae.txt")
        test_maes.append(mae_metrics["TEST"])
        print("wait")
    # Get test data split
    data_module_xval = HelloDataModule(
        root_folder=HelloContainer().local_dataset,  # type: ignore
        cross_validation_split_index=0,
        number_of_cross_validation_splits=5)
    test_dataloader = data_module_xval.test_dataloader()
    # Run inference loop
    first_child.on_inference_start()
    first_child.on_inference_start_dataset(execution_mode=ModelExecutionMode.TEST, is_ensemble_model=True)
    for batch_idx, batch in enumerate(test_dataloader):
        posteriors = first_child.forward(batch['x'])
        first_child.record_posteriors(batch, batch_idx, posteriors)
    first_child.on_inference_end_dataset()
    mse_metrics = _load_metrics(metrics_file=first_child.file_system_config.run_folder / "test_mse.txt")
    test_mse = mse_metrics["TEST"]
    mae_metrics = _load_metrics(metrics_file=loaded_config.file_system_config.run_folder / "test_mae.txt")
    test_mae = mae_metrics["TEST"]
    print("wait")

def _load_metrics(metrics_file: Path) -> Dict[str, float]:
    """
    Load the metrics for each execution mode present in the file saved during inference.
    :param metrics_file_path: The path to the metrics file saved during inference.
    :returns: MAp between the execution mode string and the metric's values found in the file.
    """
    text = metrics_file.read_text()
    metrics: Dict[str, float] = {}
    for line in text.split('\n'):
        if line:
            splits = line.split(": ")
            execution_mode_string = splits[0]
            metrics_value = float(splits[1])
            metrics[execution_mode_string] = metrics_value
    return metrics

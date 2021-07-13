#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import List
from unittest import mock

import numpy as np
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
    runner = default_runner()
    local_dataset = test_output_dirs.root_dir / "dataset"
    local_dataset.mkdir()
    args = ["", "--model=DummyEnsembleRegressionContainer", "--model_configs_namespace=Tests.ML.configs",
            f"--output_to={test_output_dirs.root_dir}", f"--local_dataset={local_dataset}",
            "--number_of_cross_validation_splits=0"]
    # Since ross validation for Lightning models will not run locally in our infrastructure we need to set up the data
    # manually for each of the cross validation runs.
    raw_data = np.loadtxt(
        tests_root_directory().parent() / "InnerEye" / "ML" / "other" / "hellocontainer.csv",
        delimiter=",")
    np.random.seed(42)
    np.random.shuffle(raw_data)
    test_data = raw_data[70:100]
    raw_data_remaining = raw_data[0:70]
    k_fold = KFold(n_splits=5)
    for cross_validation_split_index in range(5):
        train_indexes, val_indexes = list(k_fold.split(raw_data_remaining))[cross_validation_split_index]
        train_data = raw_data_remaining[train_indexes]
        val_data = raw_data_remaining[val_indexes]
        fold_data = np.append(train_data, [val_data, test_data])
        fold_data.wr
        with mock.patch("sys.argv", args):
            loaded_config, actual_run = runner.run()
    # Test if the outputs folder is relative to the folder that we specified via the commandline
    runner.lightning_container.outputs_folder.relative_to(test_output_dirs.root_dir)
    results = runner.lightning_container.outputs_folder
    # Test if all the files that are written during inference exist.
    assert not (results / "on_inference_start.txt").is_file()
    assert (results / "test_step.txt").is_file()

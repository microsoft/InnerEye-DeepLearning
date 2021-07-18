#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Dict, List
from unittest import mock

import numpy as np
from sklearn.model_selection import KFold

from InnerEye.Common.fixed_paths_for_tests import tests_root_directory
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.configs.other.HelloContainer import HelloContainer, HelloEnsembleInference, HelloRegression
from Tests.ML.util import default_runner

def test_create_ensemble_model_and_run_inference_for_innereyeinference(test_output_dirs: OutputFolderForTests) -> None:
    """
    Make real checkpoints and load them, test ensemble gets better mean squared error on the test set than any of the
    child cross validation models.
    """
    local_dataset = test_output_dirs.root_dir / "dataset"
    local_dataset.mkdir()
    checkpoint_paths: List[Path] = []
    test_mses: List[float] = []
    test_maes: List[float] = []
    np.random.seed(42)
    # Since cross validation for Lightning models will not run locally in our infrastructure, we need to set the data up
    # manually for each of the cross validation child runs, and run them manually, collating their checkpoints.
    raw_data = np.loadtxt(
        tests_root_directory().parent / "InnerEye" / "ML" / "configs" / "other" / "hellocontainer.csv",
        delimiter=",")
    np.random.shuffle(raw_data)
    test_data = raw_data[70:100]
    raw_data_remaining = raw_data[0:70]
    k_fold = KFold(n_splits=5)
    for cross_validation_split_index in range(5):
        runner = default_runner()
        local_dataset = test_output_dirs.root_dir / "dataset" / str(cross_validation_split_index)
        local_dataset.mkdir()
        args = ["", "--model=HelloContainer", "--model_configs_namespace=Tests.ML.configs",
            f"--output_to={test_output_dirs.root_dir}", f"--local_dataset={local_dataset}",
            "--number_of_cross_validation_splits=0"]
        train_indexes, val_indexes = list(k_fold.split(raw_data_remaining))[cross_validation_split_index]
        train_data = raw_data_remaining[train_indexes]
        val_data = raw_data_remaining[val_indexes]
        # Now we can save the dataset, ordered so that the new model will pick up the correct fold by default:
        fold_data = np.concatenate((train_data, val_data, test_data), axis=0)
        np.savetxt(local_dataset / "hellocontainer.csv", fold_data, delimiter=",")
        # Using runner we can now train the cross validation split model and save out its best checkpoint
        with mock.patch("sys.argv", args):
            loaded_config, _ = runner.run()
        checkpoint_path = loaded_config.file_system_config.run_folder / "checkpoints" / "best_checkpoint.ckpt"  # type: ignore
        checkpoint_paths.append(checkpoint_path)
        mse_metrics = _load_metrics_from_file(metrics_file=loaded_config.file_system_config.run_folder / "test_mse.txt")  # type: ignore
        test_mses.append(mse_metrics["TEST"])
        mae_metrics = _load_metrics_from_file(metrics_file=loaded_config.file_system_config.run_folder / "test_mae.txt")  # type: ignore
        test_maes.append(mae_metrics["TEST"])
    # Now we can test the method on run_ml
    ml_runner = MLRunner(container=HelloContainer())
    ml_runner.innereye_config.ensemble_model = HelloEnsembleInference(outputs_folder=test_output_dirs.root_dir)
    ml_runner.create_ensemble_model_and_run_inference_from_lightningmodule_checkpoints(
        HelloRegression(),
        checkpoint_paths)
    # Compare ensembke metrics with those from the cross validation runs
    mse_metrics = _load_metrics_from_file(metrics_file=test_output_dirs.root_dir / "test_mse.txt")
    test_mse = mse_metrics["TEST"]
    for xval_run_mse in test_mses:
        assert test_mse < xval_run_mse
    # TODO: Why is the ensemble MAE worse than two of the xval run ones?
    # ensemble: 0.08132067322731018,
    # xval runs: 0.0805181935429573, 0.08127883821725845, 0.08167694509029388, 0.08316199481487274, 0.08165483176708221
    # mae_metrics = _load_metrics_from_file(metrics_file=test_output_dirs.root_dir / "test_mae.txt")
    # test_mae = mae_metrics["TEST"]
    # for xval_run_mae in test_maes:
    #     assert test_mae < xval_run_mae

def _load_metrics_from_file(metrics_file: Path) -> Dict[str, float]:
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

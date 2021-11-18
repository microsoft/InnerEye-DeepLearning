#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from pathlib import Path
from typing import Dict
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from pl_bolts.models.self_supervised.resnets import ResNet
from torch.optim.lr_scheduler import _LRScheduler

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import is_windows
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLDatasetName
from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye
from InnerEye.ML.SSL.lightning_modules.simclr_module import SimCLRInnerEye
from InnerEye.ML.SSL.lightning_modules.ssl_classifier_module import SSLClassifier
from InnerEye.ML.SSL.lightning_modules.ssl_online_evaluator import EVALUATOR_STATE_NAME, OPTIMIZER_STATE_NAME, \
    SSLOnlineEvaluatorInnerEye
from InnerEye.ML.SSL.utils import SSLDataModuleType, SSLTrainingType
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
from InnerEye.ML.configs.ssl.CXR_SSL_configs import CXRImageClassifier
from InnerEye.ML.runner import Runner
from Tests.ML.utils.test_io_util import write_test_dicom

path_to_test_dataset = full_ml_test_data_path("cxr_test_dataset")


def _create_test_cxr_data(path_to_test_dataset: Path) -> None:
    """
    Creates fake datasets dataframe and dicom images mimicking the expected structure of the datasets
    of NIHCXR and RSNAKaggleCXR
    :param path_to_test_dataset: folder to which we want to save the mock data.
    """
    if path_to_test_dataset.exists():
        return
    path_to_test_dataset.mkdir(exist_ok=True)
    df = pd.DataFrame({"Image Index": np.repeat("1.dcm", 200)})
    df.to_csv(path_to_test_dataset / "Data_Entry_2017.csv", index=False)
    df = pd.DataFrame({"subject": np.repeat("1", 300),
                       "label": np.random.RandomState(42).binomial(n=1, p=0.2, size=300)})
    df.to_csv(path_to_test_dataset / "dataset.csv", index=False)
    write_test_dicom(array=np.ones([256, 256], dtype="uint16"), path=path_to_test_dataset / "1.dcm")


def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


common_test_args = ["", "--is_debug_model=True", "--num_epochs=1", "--ssl_training_batch_size=10",
                    "--linear_head_batch_size=5",
                    "--num_workers=0"]


def _compare_stored_metrics(runner: Runner, expected_metrics: Dict[str, float], abs: float = 1e-5) -> None:
    """
    Checks if the StoringLogger in the given runner holds all the expected metrics as results of training
    epoch 0, up to a given absolute precision.
    :param runner: The Innereye runner.
    :param expected_metrics: A dictionary with all metrics that are expected to be present.
    """
    assert runner.ml_runner is not None
    assert runner.ml_runner.storing_logger is not None
    print(f"Actual metrics in epoch 0: {runner.ml_runner.storing_logger.results_per_epoch[0]}")
    print(f"Expected metrics: {expected_metrics}")
    for metric, expected in expected_metrics.items():
        actual = runner.ml_runner.storing_logger.results_per_epoch[0][metric]
        if isinstance(actual, float):
            if math.isnan(expected):
                assert math.isnan(actual), f"Metric {metric}: Expected NaN, but got: {actual}"
            else:
                assert actual == pytest.approx(expected, abs=abs), f"Mismatch for metric {metric}"
        else:
            assert actual == expected, f"Mismatch for metric {metric}"


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_innereye_ssl_container_cifar10_resnet_simclr() -> None:
    """
    Tests:
        - training of SSL model on cifar10 for one epoch
        - checkpoint saving
        - checkpoint loading and ImageClassifier module creation
        - training of image classifier for one epoch.
    """
    args = common_test_args + ["--model=CIFAR10SimCLR"]
    runner = default_runner()
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert loaded_config is not None
    assert isinstance(loaded_config.model, SimCLRInnerEye)
    assert loaded_config.encoder_output_dim == 2048
    assert loaded_config.l_rate == 1e-4
    assert loaded_config.num_epochs == 1
    assert loaded_config.recovery_checkpoint_save_interval == 200
    assert loaded_config.ssl_training_type == SSLTrainingType.SimCLR
    assert loaded_config.online_eval.num_classes == 10
    assert loaded_config.online_eval.dataset == SSLDatasetName.CIFAR10.value
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10
    assert not loaded_config.use_balanced_binary_loss_for_linear_head
    assert isinstance(loaded_config.model.encoder.cnn_model, ResNet)

    # Check the metrics that were recorded during training
    expected_metrics = {
        'simclr/train/loss': 3.423144578933716,
        'simclr/learning_rate': 0.0,
        'ssl_online_evaluator/train/loss': 2.6143882274627686,
        'ssl_online_evaluator/train/online_AccuracyAtThreshold05': 0.0,
        'epoch_started': 0.0,
        'simclr/val/loss': 2.886892795562744,
        'ssl_online_evaluator/val/loss': 2.2472469806671143,
        'ssl_online_evaluator/val/AccuracyAtThreshold05': 0.20000000298023224
    }
    _compare_stored_metrics(runner, expected_metrics, abs=5e-5)

    # Check that the checkpoint contains both the optimizer for the embedding and for the linear head
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    checkpoint = torch.load(checkpoint_path)
    assert len(checkpoint["optimizer_states"]) == 1
    assert len(checkpoint["lr_schedulers"]) == 1
    assert "callbacks" in checkpoint
    assert SSLOnlineEvaluatorInnerEye in checkpoint["callbacks"]
    callback_state = checkpoint["callbacks"][SSLOnlineEvaluatorInnerEye]
    assert OPTIMIZER_STATE_NAME in callback_state
    assert EVALUATOR_STATE_NAME in callback_state

    # Now run the actual SSL classifier off the stored checkpoint
    args = common_test_args + ["--model=SSLClassifierCIFAR", f"--local_ssl_weights_path={checkpoint_path}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    assert loaded_config is not None
    assert isinstance(loaded_config.model, SSLClassifier)
    assert loaded_config.model.class_weights is None
    assert loaded_config.model.num_classes == 10


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_load_innereye_ssl_container_cifar10_cifar100_resnet_byol() -> None:
    """
    Tests that the parameters feed into the BYOL model and online evaluator are
    indeed the one we fed through our command line args
    """
    args = common_test_args + ["--model=CIFAR10CIFAR100BYOL"]
    runner = default_runner()
    with mock.patch("sys.argv", args):
        runner.parse_and_load_model()
    loaded_config = runner.lightning_container
    assert loaded_config is not None
    assert loaded_config.linear_head_dataset_name == SSLDatasetName.CIFAR100
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10
    assert loaded_config.ssl_training_type == SSLTrainingType.BYOL


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_innereye_ssl_container_rsna() -> None:
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    _create_test_cxr_data(path_to_test_dataset)
    # Test training of SSL model
    args = common_test_args + ["--model=NIH_RSNA_BYOL",
                               f"--local_dataset={str(path_to_test_dataset)}",
                               f"--extra_local_dataset_paths={str(path_to_test_dataset)}",
                               "--use_balanced_binary_loss_for_linear_head=True",
                               f"--ssl_encoder={EncoderName.densenet121.value}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert loaded_config is not None
    assert isinstance(loaded_config.model, BYOLInnerEye)
    assert loaded_config.online_eval.dataset == SSLDatasetName.RSNAKaggleCXR.value
    assert loaded_config.online_eval.num_classes == 2
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.NIHCXR
    assert loaded_config.ssl_training_type == SSLTrainingType.BYOL
    assert loaded_config.encoder_output_dim == 1024  # DenseNet output size
    # Check model params
    assert isinstance(loaded_config.model.hparams, Dict)
    assert loaded_config.model.hparams["batch_size"] == 10
    assert loaded_config.model.hparams["use_7x7_first_conv_in_resnet"]
    assert loaded_config.model.hparams["encoder_name"] == EncoderName.densenet121.value
    assert loaded_config.model.hparams["learning_rate"] == 1e-4
    assert loaded_config.model.hparams["num_samples"] == 180

    # Check some augmentation params
    assert loaded_config.datamodule_args[
               SSLDataModuleType.ENCODER].augmentation_params.preprocess.center_crop_size == 224
    assert loaded_config.datamodule_args[SSLDataModuleType.ENCODER].augmentation_params.augmentation.use_random_crop
    assert loaded_config.datamodule_args[SSLDataModuleType.ENCODER].augmentation_params.augmentation.use_random_affine

    expected_metrics = {
        'byol/train/loss': 0.00401744619011879,
        'byol/tau': 0.9899999499320984,
        'byol/learning_rate/0/0': 0.0,
        'byol/learning_rate/0/1': 0.0,
        'ssl_online_evaluator/train/loss': 0.685592532157898,
        'ssl_online_evaluator/train/online_AreaUnderRocCurve': 0.5,
        'ssl_online_evaluator/train/online_AreaUnderPRCurve': 0.699999988079071,
        'ssl_online_evaluator/train/online_AccuracyAtThreshold05': 0.4000000059604645,
        'epoch_started': 0.0,
        'byol/val/loss': -0.07644838094711304,
        'ssl_online_evaluator/val/loss': 0.6965796947479248,
        'ssl_online_evaluator/val/AreaUnderRocCurve': math.nan,
        'ssl_online_evaluator/val/AreaUnderPRCurve': math.nan,
        'ssl_online_evaluator/val/AccuracyAtThreshold05': 0.0
    }
    _compare_stored_metrics(runner, expected_metrics)

    # Check that we are able to load the checkpoint and create classifier model
    checkpoint_path = loaded_config.checkpoint_folder / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    args = common_test_args + ["--model=CXRImageClassifier",
                               f"--local_dataset={str(path_to_test_dataset)}",
                               "--use_balanced_binary_loss_for_linear_head=True",
                               f"--local_ssl_weights_path={checkpoint_path}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert loaded_config is not None
    assert isinstance(loaded_config, CXRImageClassifier)
    assert loaded_config.model.freeze_encoder
    assert torch.isclose(loaded_config.model.class_weights, torch.tensor([0.21, 0.79]), atol=1e-6).all()  # type: ignore
    assert loaded_config.model.num_classes == 2


def test_simclr_lr_scheduler() -> None:
    """
    Test if the LR scheduler has the expected warmup behaviour.
    """
    num_samples = 100
    batch_size = 20
    gpus = 1
    max_epochs = 10
    warmup_epochs = 2
    model = SimCLRInnerEye(encoder_name="resnet18", dataset_name="CIFAR10",
                           gpus=gpus, num_samples=num_samples, batch_size=batch_size,
                           max_epochs=max_epochs, warmup_epochs=warmup_epochs)
    # The LR scheduler used here works per step. Scheduler computes the total number of steps, in this example that's 5
    train_iters_per_epoch = num_samples / (batch_size * gpus)
    assert model.train_iters_per_epoch == train_iters_per_epoch
    # Mock a second optimizer that is normally created in the SSL container
    linear_head_optimizer = mock.MagicMock()
    model.online_eval_optimizer = linear_head_optimizer
    # Retrieve the scheduler and iterate it
    _, scheduler_list = model.configure_optimizers()
    assert isinstance(scheduler_list[0], dict)
    assert scheduler_list[0]["interval"] == "step"
    scheduler = scheduler_list[0]["scheduler"]
    assert isinstance(scheduler, _LRScheduler)
    lr = []
    for i in range(0, int(max_epochs * train_iters_per_epoch)):
        scheduler.step()
        lr.append(scheduler.get_last_lr()[0])
    # The highest learning rate is expected after the warmup epochs
    highest_lr = np.argmax(lr)
    assert highest_lr == int(warmup_epochs * train_iters_per_epoch - 1)

    for i in range(0, highest_lr):
        assert lr[i] < lr[i + 1], f"Not strictly monotonically increasing at index {i}"
    for i in range(highest_lr, len(lr) - 1):
        assert lr[i] > lr[i + 1], f"Not strictly monotonically decreasing at index {i}"

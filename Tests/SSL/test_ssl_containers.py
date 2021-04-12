#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from unittest import mock

import numpy as np
import pytest
from pl_bolts.models.self_supervised.resnets import ResNet

from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.ML.runner import Runner
from InnerEye.SSL.byol.byol_module import WrapBYOLInnerEye
from InnerEye.SSL.encoders import DenseNet121Encoder
from InnerEye.SSL.lightning_containers.ssl_container import EncoderName, SSLDatasetName
from InnerEye.SSL.simclr_module import WrapSimCLRInnerEye
from InnerEye.SSL.utils import SSLModule, SSLType


def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


common_test_args = ["", "--debug=True", "--num_epochs=1", "--ssl_training_batch_size=10", "--classifier_batch_size=5",
                    "--num_workers=0"]


def test_innereye_ssl_container_cifar10_resnet_simclr() -> None:
    """
    Tests:
        - training of SSL model on cifar10 for one epoch
        - checkpoint saving
        - checkpoint loading and ImageClassifier module creation
        - training of image classifier for one epoch.
    """
    args = common_test_args + ["--model=CIFAR10SimCLR"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    assert loaded_config.encoder_output_dim == 2048
    assert loaded_config.l_rate == 1e-4
    assert isinstance(loaded_config.model, WrapSimCLRInnerEye)
    assert loaded_config.num_epochs == 1
    assert loaded_config.recovery_checkpoint_save_interval == 200
    assert loaded_config.ssl_training_type == SSLType.SimCLR
    assert loaded_config.use_balanced_binary_loss_for_linear_head == False
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    args = common_test_args + ["--model=SSLClassifierCIFAR", f"--local_ssl_weights_path={checkpoint_path}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()


def test_innereye_ssl_container_cifar10_resnet_byol() -> None:
    args = common_test_args + ["--model=CIFAR10BYOL"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    assert loaded_config.ssl_training_type == SSLType.BYOL
    assert isinstance(loaded_config.model, WrapBYOLInnerEye)
    assert loaded_config.online_eval.num_classes == 10
    assert loaded_config.online_eval.dataset == SSLDatasetName.CIFAR10.value
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10


def test_innereye_ssl_container_cifar10_cifar100_resnet_byol() -> None:
    """
    Tests that the parameters feed into the BYOL model and online evaluator are
    indeed the one we fed through our command line args
    """
    args = common_test_args + ["--model=CIFAR10CIFAR100BYOL"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    assert loaded_config.online_eval.dataset == SSLDatasetName.CIFAR100.value
    assert loaded_config.online_eval.num_classes == 100
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10
    assert loaded_config.ssl_training_type == SSLType.BYOL
    assert loaded_config.model.hparams["batch_size"] == 10
    assert loaded_config.model.hparams["dataset_name"] == SSLDatasetName.CIFAR10.value
    assert loaded_config.model.hparams["encoder_name"] == EncoderName.resnet50.value
    assert loaded_config.model.hparams["learning_rate"] == 1e-4
    assert isinstance(loaded_config.model.online_network.encoder.cnn_model, ResNet)

def test_innereye_ssl_container_cifar10_densenet() -> None:
    args = common_test_args + ["--model=CIFAR10BYOL", f"--ssl_encoder={EncoderName.densenet121.value}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    assert isinstance(loaded_config.model.online_network.encoder.cnn_model, DenseNet121Encoder)
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    assert checkpoint_path.exists()


@pytest.mark.parametrize("use_binary_loss_linear_head", [True, False])
def test_innereye_ssl_container_rsna(use_binary_loss_linear_head: bool):
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    path_to_test_dataset = str(repository_root_directory() / "Tests" / "SSL" / "test_dataset")
    args = common_test_args + ["--model=RSNA_RSNA_BYOL", f"--local_dataset={path_to_test_dataset}",
                               f"--use_balanced_binary_loss_for_linear_head={use_binary_loss_linear_head}"]
    with mock.patch("sys.argv", args), mock.patch(
            'InnerEye.SSL.datamodules.cxr_datasets.InnerEyeCXRDatasetBase.read_dicom',
            return_value=np.ones([256, 256])):
        loaded_config, actual_run = runner.run()
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    args = common_test_args + ["--model=CXRImageClassifier", f"--extra_local_dataset_paths={path_to_test_dataset}",
                               f"--use_balanced_binary_loss_for_linear_head={use_binary_loss_linear_head}",
                               f"--local_ssl_weights_path={checkpoint_path}"]
    with mock.patch("sys.argv", args), mock.patch(
            'InnerEye.SSL.datamodules.cxr_datasets.InnerEyeCXRDatasetBase.read_dicom',
            return_value=np.ones([256, 256])):
        loaded_config, actual_run = runner.run()
    assert loaded_config.online_eval.dataset == SSLDatasetName.RSNAKaggle.value
    assert loaded_config.online_eval.num_classes == 2
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.RSNAKaggle
    assert loaded_config.ssl_training_type == SSLType.BYOL
    assert loaded_config.encoder_output_dim == 2048
    # Check model params
    assert loaded_config.model.hparams["batch_size"] == 10
    assert loaded_config.model.hparams["dataset_name"] == SSLDatasetName.RSNAKaggle.value
    assert loaded_config.model.hparams["encoder_name"] == EncoderName.resnet50.value
    assert loaded_config.model.hparams["learning_rate"] == 1e-4
    assert loaded_config.model.hparams["num_samples"] == 240
    # Check some augmentation params
    assert loaded_config.datamodule_args[SSLModule.ENCODER]["augmentation_config"].preprocess.center_crop_size == 224
    assert loaded_config.datamodule_args[SSLModule.ENCODER]["augmentation_config"].augmentation.use_random_crop
    assert loaded_config.datamodule_args[SSLModule.ENCODER]["augmentation_config"].augmentation.use_random_affine

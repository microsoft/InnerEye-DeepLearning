#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Dict
from unittest import mock

import numpy as np
import pytest
import torch
from pl_bolts.models.self_supervised.resnets import ResNet

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import is_windows
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.ML.SSL.byol.byol_module import BYOLInnerEye
from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLDatasetName
from InnerEye.ML.SSL.lightning_containers.ssl_image_classifier import SSLClassifier
from InnerEye.ML.SSL.simclr_module import SimCLRInnerEye
from InnerEye.ML.SSL.utils import SSLModule, SSLType
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
from InnerEye.ML.configs.ssl.CXR_SSL_configs import CXRImageClassifier
from InnerEye.ML.runner import Runner
from Tests.ML.utils.test_io_util import write_test_dicom


def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


common_test_args = ["", "--debug=True", "--num_epochs=1", "--ssl_training_batch_size=10", "--classifier_batch_size=5",
                    "--num_workers=0"]


# @pytest.mark.skipif(is_windows(), reason="Too slow on windows")
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
    assert loaded_config is not None
    assert isinstance(loaded_config.model, SimCLRInnerEye)
    assert loaded_config.encoder_output_dim == 2048
    assert loaded_config.l_rate == 1e-4
    assert loaded_config.num_epochs == 1
    assert loaded_config.recovery_checkpoint_save_interval == 200
    assert loaded_config.ssl_training_type == SSLType.SimCLR
    assert loaded_config.online_eval.num_classes == 10
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10
    assert loaded_config.online_eval.dataset == SSLDatasetName.CIFAR10.value
    assert not loaded_config.use_balanced_binary_loss_for_linear_head
    assert isinstance(loaded_config.model.encoder.cnn_model, ResNet)
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
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
    assert loaded_config.classifier_dataset_name == SSLDatasetName.CIFAR100
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.CIFAR10
    assert loaded_config.ssl_training_type == SSLType.BYOL


# @pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_innereye_ssl_container_rsna() -> None:
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    path_to_test_dataset = repository_root_directory() / "Tests" / "SSL" / "test_dataset"
    write_test_dicom(array=np.ones([256, 256], dtype="uint16"), path=path_to_test_dataset / "1.dcm")

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
    assert loaded_config.online_eval.dataset == SSLDatasetName.RSNAKaggle.value
    assert loaded_config.online_eval.num_classes == 2
    assert loaded_config.ssl_training_dataset_name == SSLDatasetName.NIH
    assert loaded_config.ssl_training_type == SSLType.BYOL
    assert loaded_config.encoder_output_dim == 1024  # DenseNet output size
    # Check model params
    assert isinstance(loaded_config.model.hparams, Dict)
    assert loaded_config.model.hparams["batch_size"] == 10
    assert loaded_config.model.hparams["dataset_name"] == SSLDatasetName.NIH.value
    assert loaded_config.model.hparams["encoder_name"] == EncoderName.densenet121.value
    assert loaded_config.model.hparams["learning_rate"] == 1e-4
    assert loaded_config.model.hparams["num_samples"] == 270

    # Check some augmentation params
    assert loaded_config.datamodule_args[SSLModule.ENCODER].augmentation_config.preprocess.center_crop_size == 224
    assert loaded_config.datamodule_args[SSLModule.ENCODER].augmentation_config.augmentation.use_random_crop
    assert loaded_config.datamodule_args[SSLModule.ENCODER].augmentation_config.augmentation.use_random_affine

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
    assert torch.isclose(loaded_config.model.class_weights, torch.tensor([0.2, 0.8]), atol=1e-6).all()  # type: ignore
    assert loaded_config.model.num_classes == 2

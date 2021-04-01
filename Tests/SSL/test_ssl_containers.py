#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from unittest import mock
import numpy as np
import pytest

from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.ML.runner import Runner

# todo container with linear head is missing. Implement and re-add to test to recover train & recover
# todo functionality


def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


def test_innereye_ssl_container_cifar10_resnet_simclr() -> None:
    args = ["", "--model=CIFAR10SimCLR", "--debug=True", "--num_epochs=1", "--batch_size=20"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()


def test_innereye_ssl_container_cifar10_resnet_byol() -> None:
    args = ["", "--model=CIFAR10BYOL", "--debug=True", "--num_epochs=1", "--batch_size=20"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()


def test_innereye_ssl_container_cifar10_cifar100_resnet_byol() -> None:
    args = ["", "--model=CIFAR10CIFAR100BYOL", "--debug=True", "--num_epochs=1", "--batch_size=20"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()



def test_innereye_ssl_container_cifar10_densenet() -> None:
    args = ["", "--model=CIFARBYOL", "--debug=True", "--num_epochs=2", "--ssl_encoder=densenet121"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    assert checkpoint_path.exists()


@pytest.mark.parametrize("use_binary_loss_linear_head", [True, False])
def test_innereye_ssl_container_rsna(use_binary_loss_linear_head: bool):
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    path_to_test_dataset = str(repository_root_directory() / "Tests" / "SSL" / "test_dataset")
    args = ["", "--model=RSNA_RSNA_BYOL", f"--local_dataset={path_to_test_dataset}", "--debug=True", "--num_epochs=1",
            "--batch_size=10", "--num_workers=0", "--linear_head_batch_size=5",
            f"--use_balanced_binary_loss_for_linear_head={use_binary_loss_linear_head}"]
    with mock.patch("sys.argv", args), mock.patch(
            'InnerEye.SSL.datamodules.cxr_datasets.InnerEyeCXRDatasetBase.read_dicom',
            return_value=np.ones([256, 256])):
        loaded_config, actual_run = runner.run()

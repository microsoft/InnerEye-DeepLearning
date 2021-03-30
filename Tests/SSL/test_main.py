#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from random import randint
from unittest import mock

import torch

from InnerEye.Common import fixed_paths
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.ML.runner import Runner
from InnerEye.SSL.datamodules.rsna_cxr_dataset import RSNAKaggleCXR

def default_runner() -> Runner:
    """
    Create an InnerEye Runner object with the default settings, pointing to the repository root and
    default settings files.
    """
    return Runner(project_root=repository_root_directory(),
                  yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)


def test_innereye_ssl_container_cifar10_resnet_simclr() -> None:
    args = ["", "--model=CIFARBYOL", "--debug=True", "--num_epochs=2"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()


# todo container with linear head is missing. Implement and re-add to test to recover train & recover
# todo functionality
def test_innereye_ssl_container_cifar10_densenet() -> None:
    args = ["", "--model=CIFARBYOL", "--debug=True", "--num_epochs=2", "--ssl_encoder=densenet121"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    checkpoint_path = loaded_config.outputs_folder / "checkpoints" / "best_checkpoint.ckpt"
    assert checkpoint_path.exists()
    """
    args = ["", "--model=DummyLinearImageClassifier", f"--path_yaml_config={str(path_to_config)}",
            "--model_configs_namespace=Tests.ML.configs", f"--local_weights_path={checkpoint_path}"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = default_runner().run()
    """


def _get_dummy_val_train_rsna_dataloaders():
    """
    Return dummy train and validation datasets
    """
    dataset_dir = str(Path(__file__).parent / "test_dataset")

    class DummyRSNADataset(RSNAKaggleCXR):
        def __getitem__(self, item):
            return (torch.rand([3, 224, 224], dtype=torch.float32),
                    torch.rand([3, 224, 224], dtype=torch.float32)), \
                   randint(0, 1)

    dummy_rsna_train_dataloader = torch.utils.data.DataLoader(
        DummyRSNADataset(dataset_dir, True),
        batch_size=20,
        num_workers=0,
        drop_last=True)
    dummy_rsna_val_dataloader = torch.utils.data.DataLoader(
        DummyRSNADataset(dataset_dir, False),
        batch_size=20,
        num_workers=0,
        drop_last=True)
    return dummy_rsna_train_dataloader, dummy_rsna_val_dataloader


# todo test both cases with and with binary loss
def test_innereye_ssl_container_rsna():
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = default_runner()
    path_to_test_dataset = str(repository_root_directory() / "Tests" / "SSL" / "test_dataset")
    path_to_config = repository_root_directory() / "InnerEye" / "ML" / "configs"/ "ssl"/ "rsna_augmentations.yaml"
    args = ["", "--model=RSNAKaggleBYOL", f"--path_augmentation_config={str(path_to_config)}",
            f"--local_dataset={path_to_test_dataset}", "--debug=True", "--num_epochs=1", "--batch_size=25"]
    with mock.patch("sys.argv", args):
        dummy_rsna_train_dataloader, dummy_rsna_val_dataloader = _get_dummy_val_train_rsna_dataloaders()
        with mock.patch("InnerEye.SSL.datamodules.chestxray_datamodule.RSNAKaggleDataModule.train_dataloader",
                        return_value=dummy_rsna_train_dataloader):
            with mock.patch("InnerEye.SSL.datamodules.chestxray_datamodule.RSNAKaggleDataModule.val_dataloader",
                            return_value=dummy_rsna_val_dataloader):
                loaded_config, actual_run = runner.run()

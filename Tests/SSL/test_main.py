#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from random import randint
from unittest import mock

import pytest
import torch

from InnerEye.Common.fixed_paths import SSL_EXPERIMENT_DIR, repository_root_directory
from InnerEye.SSL.datamodules.rsna_cxr_dataset import RSNAKaggleCXR
from InnerEye.SSL.main import cli_main, get_last_checkpoint_path
from InnerEye.SSL.utils import create_ssl_image_classifier, load_ssl_model_config


def test_train_and_recover_SimCLRClassifier_cifar10() -> None:
    config = load_ssl_model_config(
        repository_root_directory() / "InnerEye" / "SSL" / "configs" / "cifar10_simclr.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.batch_size = 100
    config.train.self_supervision.encoder_name = "resnet18"
    cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(SSL_EXPERIMENT_DIR / config.train.output_dir,
                                         f'simclr_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=10, pl_checkpoint_path=last_cpkt)


def test_train_and_recover_BYOLClassifier_cifar10_resnet() -> None:
    config = load_ssl_model_config(
        repository_root_directory() / "InnerEye" / "SSL" / "configs" / "cifar10_byol.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.batch_size = 100
    config.train.self_supervision.encoder_name = "resnet18"
    cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(SSL_EXPERIMENT_DIR / config.train.output_dir, f'byol_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=10, pl_checkpoint_path=last_cpkt)


def test_train_and_recover_BYOLClassifier_cifar10_densenet() -> None:
    config = load_ssl_model_config(
        repository_root_directory() / "InnerEye" / "SSL" / "configs" / "cifar10_byol.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.batch_size = 50
    config.train.self_supervision.encoder_name = "densenet121"
    cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(SSL_EXPERIMENT_DIR / config.train.output_dir, f'byol_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=10, pl_checkpoint_path=last_cpkt)


@pytest.mark.parametrize("balanced_binary_loss", [False, True])
def test_train_and_recover_BYOLClassifier_rsna_resnet(balanced_binary_loss: bool) -> None:
    config = load_ssl_model_config(
        repository_root_directory() / "InnerEye" / "SSL" / "configs" / "rsna_byol.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.self_supervision.encoder_name = "resnet18"
    config.dataset.dataset_dir = str(Path(__file__).parent / "test_dataset")
    config.train.self_supervision.use_balanced_binary_loss_for_linear_head = balanced_binary_loss
    dummy_rsna_train_dataloader, dummy_rsna_val_dataloader = _get_dummy_val_train_rsna_dataloaders(
        config.dataset.dataset_dir)
    with mock.patch("InnerEye.SSL.datamodules.chestxray_datamodule.RSNAKaggleDataModule.train_dataloader",
                    return_value=dummy_rsna_train_dataloader):
        with mock.patch("InnerEye.SSL.datamodules.chestxray_datamodule.RSNAKaggleDataModule.val_dataloader",
                        return_value=dummy_rsna_val_dataloader):
            cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(SSL_EXPERIMENT_DIR / config.train.output_dir, f'byol_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=2, pl_checkpoint_path=last_cpkt)


def _get_dummy_val_train_rsna_dataloaders(dataset_dir: Path):
    """
    Return dummy train and validation datasets
    """

    class DummyRSNADataset(RSNAKaggleCXR):
        def __getitem__(self, item):
            return (torch.rand([3, 224, 224], dtype=torch.float32), torch.rand([3, 224, 224], dtype=torch.float32),
                    torch.rand([3, 224, 224])), randint(0, 1)

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

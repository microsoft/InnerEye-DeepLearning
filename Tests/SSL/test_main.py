#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.fixed_paths import SSL_EXPERIMENT_DIR, repository_root_directory
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
    last_cpkt = get_last_checkpoint_path(SSL_EXPERIMENT_DIR / config.train.output_dir, f'simclr_seed_{config.train.seed}')
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

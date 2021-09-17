#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch

from pathlib import Path

from default_paths import EXPERIMENT_DIR
from InnerEyeDataQuality.deep_learning.self_supervised.main import cli_main, get_last_checkpoint_path
from InnerEyeDataQuality.deep_learning.self_supervised.utils import create_ssl_image_classifier
from InnerEyeDataQuality.deep_learning.utils import load_ssl_model_config
from InnerEyeDataQuality.deep_learning.collect_embeddings import register_embeddings_collector, get_all_embeddings


repo_root = Path(__file__).parent.parent.parent.parent


def _run_inference_check_embeddings(model: torch.nn.Module) -> None:
    # Run inference on a random image and check embeddings' l2 norm is one.
    all_model_cnn_embeddings = register_embeddings_collector([model], use_only_in_train=False)
    device = next(model.parameters()).device
    model(torch.rand((2, 3, 128, 128)).to(device))

    embs = torch.from_numpy(get_all_embeddings(all_model_cnn_embeddings)[0])
    sum_l2norm = torch.sum(torch.norm(embs, p=2, dim=-1))
    print(sum_l2norm)
    print(embs.shape)
    assert torch.abs(sum_l2norm - embs.shape[0]) < 1e-5


def test_train_and_recover_SimCLRClassifier_cifar10h() -> None:
    config = load_ssl_model_config(
        repo_root / "InnerEyeDataQuality" / "deep_learning" / "self_supervised" / "configs" / "cifar10h_simclr.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.batch_size = 100
    config.train.self_supervision.encoder_name = "resnet18"
    cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(EXPERIMENT_DIR / config.train.output_dir, f'simclr_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=10, pl_checkpoint_path=last_cpkt)
    _run_inference_check_embeddings(ssl_classifier)


def test_train_and_recover_BYOLClassifier_cifar10h_resnet() -> None:
    config = load_ssl_model_config(
        repo_root / "InnerEyeDataQuality" / "deep_learning" / "self_supervised" / "configs" / "cifar10h_byol.yaml")
    config.defrost()
    config.scheduler.epochs = 1
    config.train.batch_size = 100
    config.train.self_supervision.encoder_name = "resnet18"
    cli_main(config, debug=True)
    last_cpkt = get_last_checkpoint_path(EXPERIMENT_DIR / config.train.output_dir, f'byol_seed_{config.train.seed}')
    ssl_classifier = create_ssl_image_classifier(num_classes=10, pl_checkpoint_path=last_cpkt)
    _run_inference_check_embeddings(ssl_classifier)

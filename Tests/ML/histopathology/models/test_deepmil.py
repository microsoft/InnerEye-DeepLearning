#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from typing import Callable, Dict, List, Optional, Type  # noqa

import pytest
import torch
from torch import Tensor, argmax, nn, rand, randint, randn, round, stack, allclose
from torchvision.models import resnet18

from health_ml.networks.layers.attention_layers import (
    AttentionLayer,
    GatedAttentionLayer,
    MeanPoolingLayer,
)

from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.configs.histo_configs.classification.DeepSMILECrck import (
    DeepSMILECrck,
)
from InnerEye.ML.configs.histo_configs.classification.DeepSMILEPanda import (
    DeepSMILEPanda,
)
from InnerEye.ML.Histopathology.datamodules.base_module import TilesDataModule
from InnerEye.ML.Histopathology.datasets.default_paths import (
    TCGA_CRCK_DATASET_DIR,
    PANDA_TILES_DATASET_DIR,
)
from InnerEye.ML.Histopathology.models.deepmil import DeepMILModule
from InnerEye.ML.Histopathology.models.encoders import ImageNetEncoder, TileEncoder
from InnerEye.ML.Histopathology.utils.naming import MetricsKey, ResultsKey


def get_supervised_imagenet_encoder() -> TileEncoder:
    return ImageNetEncoder(feature_extraction_model=resnet18, tile_size=224)


def _test_lightningmodule(
    n_classes: int,
    pooling_layer: Callable[[int, int, int], nn.Module],
    batch_size: int,
    max_bag_size: int,
    pool_hidden_dim: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:

    assert n_classes > 0

    # hard-coded here to avoid test explosion; correctness of other encoders is tested elsewhere
    encoder = get_supervised_imagenet_encoder()
    module = DeepMILModule(
        encoder=encoder,
        label_column="label",
        n_classes=n_classes,
        pooling_layer=pooling_layer,
        pool_hidden_dim=pool_hidden_dim,
        pool_out_dim=pool_out_dim,
        dropout_rate=dropout_rate,
    )

    bag_images = rand([batch_size, max_bag_size, *module.encoder.input_dim])
    bag_labels_list = []
    bag_logits_list = []
    bag_attn_list = []
    for bag in bag_images:
        if n_classes > 1:
            labels = randint(n_classes, size=(max_bag_size,))
        else:
            labels = randint(n_classes + 1, size=(max_bag_size,))
        bag_labels_list.append(module.get_bag_label(labels))
        logit, attn = module(bag)
        assert logit.shape == (1, n_classes)
        assert attn.shape == (module.pool_out_dim, max_bag_size)
        bag_logits_list.append(logit.view(-1))
        bag_attn_list.append(attn)

    bag_logits = stack(bag_logits_list)
    bag_labels = stack(bag_labels_list).view(-1)

    assert bag_logits.shape[0] == (batch_size)
    assert bag_labels.shape[0] == (batch_size)

    if module.n_classes > 1:
        loss = module.loss_fn(bag_logits, bag_labels)
    else:
        loss = module.loss_fn(bag_logits.squeeze(1), bag_labels.float())

    assert loss > 0
    assert loss.shape == ()

    probs = module.activation_fn(bag_logits)
    assert ((probs >= 0) & (probs <= 1)).all()
    if n_classes > 1:
        assert probs.shape == (batch_size, n_classes)
    else:
        assert probs.shape[0] == batch_size

    if n_classes > 1:
        preds = argmax(probs, dim=1)
    else:
        preds = round(probs)
    assert preds.shape[0] == batch_size

    for metric_name, metric_object in module.train_metrics.items():
        if metric_name == MetricsKey.CONF_MATRIX:
            continue
        score = metric_object(preds.view(-1, 1), bag_labels.view(-1, 1))
        assert torch.all(score >= 0)
        assert torch.all(score <= 1)


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("pooling_layer", [AttentionLayer, GatedAttentionLayer])
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("max_bag_size", [1, 7])
@pytest.mark.parametrize("pool_hidden_dim", [1, 5])
@pytest.mark.parametrize("pool_out_dim", [1, 6])
@pytest.mark.parametrize("dropout_rate", [None, 0.5])
def test_lightningmodule_attention(
    n_classes: int,
    pooling_layer: Callable[[int, int, int], nn.Module],
    batch_size: int,
    max_bag_size: int,
    pool_hidden_dim: int,
    pool_out_dim: int,
    dropout_rate: Optional[float],
) -> None:
    _test_lightningmodule(n_classes=n_classes,
                          pooling_layer=pooling_layer,
                          batch_size=batch_size,
                          max_bag_size=max_bag_size,
                          pool_hidden_dim=pool_hidden_dim,
                          pool_out_dim=pool_out_dim,
                          dropout_rate=dropout_rate)


@pytest.mark.parametrize("n_classes", [1, 3])
@pytest.mark.parametrize("batch_size", [1, 15])
@pytest.mark.parametrize("max_bag_size", [1, 7])
@pytest.mark.parametrize("dropout_rate", [None, 0.5])
def test_lightningmodule_mean_pooling(
    n_classes: int,
    batch_size: int,
    max_bag_size: int,
    dropout_rate: Optional[float],
) -> None:
    _test_lightningmodule(n_classes=n_classes,
                          pooling_layer=MeanPoolingLayer,
                          batch_size=batch_size,
                          max_bag_size=max_bag_size,
                          pool_hidden_dim=1,
                          pool_out_dim=1,
                          dropout_rate=dropout_rate)


def move_batch_to_expected_device(batch: Dict[str, List], use_gpu: bool) -> Dict:
    device = "cuda" if use_gpu else "cpu"
    return {
        key: [
            value.to(device) if isinstance(value, Tensor) else value for value in values
        ]
        for key, values in batch.items()
    }


CONTAINER_DATASET_DIR = {
    DeepSMILEPanda: PANDA_TILES_DATASET_DIR,
    DeepSMILECrck: TCGA_CRCK_DATASET_DIR,
}


@pytest.mark.parametrize("container_type", [DeepSMILEPanda,
                                            DeepSMILECrck])
@pytest.mark.parametrize("use_gpu", [True, False])
def test_container(container_type: Type[LightningContainer], use_gpu: bool) -> None:
    dataset_dir = CONTAINER_DATASET_DIR[container_type]
    if not os.path.isdir(dataset_dir):
        pytest.skip(
            f"Dataset for container {container_type.__name__} "
            f"is unavailable: {dataset_dir}"
        )
    if container_type is DeepSMILECrck:
        container = DeepSMILECrck(encoder_type=ImageNetEncoder.__name__)
    elif container_type is DeepSMILEPanda:
        container = DeepSMILEPanda(encoder_type=ImageNetEncoder.__name__)
    else:
        container = container_type()

    container.setup()

    data_module: TilesDataModule = container.get_data_module()  # type: ignore
    data_module.max_bag_size = 10
    module = container.create_model()
    if use_gpu:
        module.cuda()

    train_data_loader = data_module.train_dataloader()
    for batch_idx, batch in enumerate(train_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        loss = module.training_step(batch, batch_idx)
        loss.retain_grad()
        loss.backward()
        assert loss.grad is not None
        assert loss.shape == ()
        assert isinstance(loss, Tensor)
        break

    val_data_loader = data_module.val_dataloader()
    for batch_idx, batch in enumerate(val_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        loss = module.validation_step(batch, batch_idx)
        assert loss.shape == ()  # noqa
        assert isinstance(loss, Tensor)
        break

    test_data_loader = data_module.test_dataloader()
    for batch_idx, batch in enumerate(test_data_loader):
        batch = move_batch_to_expected_device(batch, use_gpu)
        outputs_dict = module.test_step(batch, batch_idx)
        loss = outputs_dict[ResultsKey.LOSS]  # noqa
        assert loss.shape == ()
        assert isinstance(loss, Tensor)
        break


def test_class_weights_binary() -> None:
    class_weights = Tensor([0.5, 3.5])
    n_classes = 1
    module = DeepMILModule(
        encoder=get_supervised_imagenet_encoder(),
        label_column="label",
        n_classes=n_classes,
        pooling_layer=AttentionLayer,
        pool_hidden_dim=5,
        pool_out_dim=1,
        class_weights=class_weights,
    )
    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes + 1, size=(1,))

    pos_weight = Tensor([class_weights[1] / (class_weights[0] + 1e-5)])
    loss_weighted = module.loss_fn(logits.squeeze(1), bag_label.float())
    criterion_unweighted = nn.BCEWithLogitsLoss()
    loss_unweighted = criterion_unweighted(logits.squeeze(1), bag_label.float())
    if bag_label.item() == 1:
        assert allclose(loss_weighted, pos_weight * loss_unweighted)
    else:
        assert allclose(loss_weighted, loss_unweighted)


def test_class_weights_multiclass() -> None:
    class_weights = Tensor([0.33, 0.33, 0.33])
    n_classes = 3
    module = DeepMILModule(
        encoder=get_supervised_imagenet_encoder(),
        label_column="label",
        n_classes=n_classes,
        pooling_layer=AttentionLayer,
        pool_hidden_dim=5,
        pool_out_dim=1,
        class_weights=class_weights,
    )
    logits = Tensor(randn(1, n_classes))
    bag_label = randint(n_classes, size=(1,))

    loss_weighted = module.loss_fn(logits, bag_label)
    criterion_unweighted = nn.CrossEntropyLoss()
    loss_unweighted = criterion_unweighted(logits, bag_label)
    # The weighted and unweighted loss functions give the same loss values for batch_size = 1.
    # https://stackoverflow.com/questions/67639540/pytorch-cross-entropy-loss-weights-not-working
    # TODO: the test should reflect actual weighted loss operation for the class weights after batch_size > 1 is implemented.
    assert allclose(loss_weighted, loss_unweighted)

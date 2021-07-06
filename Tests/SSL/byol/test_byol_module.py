#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest
import torch

from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye


# Test cosine loss
def test_cosine_loss() -> None:
    loss_fn = BYOLInnerEye.cosine_loss

    # Cosine loss of same tensor
    tensor_a = torch.rand((16, 128))
    loss = loss_fn(tensor_a, tensor_a)
    assert torch.isclose(loss, torch.tensor(-1.0))

    # Cosine loss between orthogonal tensors
    tensor_a = torch.Tensor([0, 1, 0, 1, 0])
    tensor_b = torch.Tensor([1, 0, 1, 0, 1])
    loss = loss_fn(tensor_a, tensor_b)
    assert torch.isclose(loss, torch.tensor(0.0))

# Test if initial set of parameters are equal between student and teacher.
def test_module_param_eq() -> None:
    byol = BYOLInnerEye(num_samples=16, learning_rate=1e-3, batch_size=4, encoder_name="resnet50", warmup_epochs=10, max_epochs=100)
    pars1 = byol.online_network.parameters()
    pars2 = byol.target_network.parameters()
    for par1, par2 in zip(pars1, pars2):
        assert torch.all(torch.eq(par1, par2))

# Test initialisation with different encoder types.
@pytest.mark.parametrize("encoder_name", ["resnet18", "resnet50", "resnet101", "densenet121"])
def test_encoder_init(encoder_name: str) -> None:
    BYOLInnerEye(num_samples=16, learning_rate=1e-3, batch_size=4, warmup_epochs=10, encoder_name=encoder_name)

# Test shared step - loss should be bounded between some value and cannot be outside that value.
def test_shared_forward_step() -> None:
    byol = BYOLInnerEye(num_samples=16, learning_rate=1e-3, batch_size=4, warmup_epochs=10, encoder_name="resnet50", max_epochs=100)
    imgs = torch.rand((4, 3, 32, 32))
    lbls = torch.rand((4,))
    batch = ([imgs, imgs], lbls)

    loss = byol.shared_step(batch=batch, batch_idx=0)
    assert torch.le(loss, 1.0)
    assert torch.ge(loss, -1.0)

# Check if output pooling works
def test_output_spatial_pooling() -> None:
    byol = BYOLInnerEye(num_samples=16, learning_rate=1e-3, batch_size=4, warmup_epochs=10, encoder_name="resnet50", max_epochs=100)
    imgs = torch.rand((4, 3, 32, 32))

    embeddings = byol(imgs)
    batch_size = embeddings.size()[0]
    embedding_size = embeddings.size()[1]

    assert batch_size == 4
    assert embedding_size == 2048

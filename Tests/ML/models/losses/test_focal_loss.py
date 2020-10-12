#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
import torch

from InnerEye.ML.models.losses.cross_entropy import CrossEntropyLoss

# Set random seed
torch.random.manual_seed(1)


def test_get_focal_loss_pixel_weights() -> None:
    """
    Weights for the correctly predicted (logits) pixels should be closer to zero,
    and wrong predictions should have higher weights. The total sum of weights should be
    equal to the number of pixels in order not to change the scale of loss function.
    """
    x_entropy_loss = CrossEntropyLoss(focal_loss_gamma=2.0)

    target = torch.tensor([[[1, 0, 0], [0, 1, 1]]], dtype=torch.float32)
    logits = torch.tensor([[[0, 0, 0], [-1e9, 0, 0]]], dtype=torch.float32)
    pixel_weights = x_entropy_loss.get_focal_loss_pixel_weights(logits=logits, target=target)
    assert torch.allclose(torch.masked_select(pixel_weights, target.eq(1.0)),
                          torch.tensor([0.00, 1.50, 1.50]))


@pytest.mark.parametrize(["use_class_balancing", "expected_loss"],
                         [(False, -1 * torch.log(torch.tensor(0.5)) / 3.0),
                          (True, -1 * torch.log(torch.tensor(0.5)) * 4.0 / 9.0)])
def test_focal_loss_forward_balanced(use_class_balancing: bool, expected_loss: torch.Tensor) -> None:
    """
    When logits are the same for both classes, cross entropy should return [0.5, 0.5] posterior probabilities.
    Loss for that particular pixel should be equal to -log(0.5) (negative log-likelihood). When loss terms are
    mean aggregated across 3 pixels or 2 classes, the result should be equal to -log(0.5)/3 and -log(0.5)/2
    respectively. Since the other two pixels are correctly predicted, their loss terms are equal to zero.
    """
    target = torch.tensor([[[0, 0, 1], [1, 1, 0]]], dtype=torch.float32)
    logits = torch.tensor([[[-1e9, -1e9, 0], [0, 0, 0]]], dtype=torch.float32)

    # Compute loss values for both balanced and unbalanced cases
    loss_fn = CrossEntropyLoss(class_weight_power=1.0 if use_class_balancing else 0.0, focal_loss_gamma=0.0)
    loss = loss_fn(logits, target)
    assert (torch.isclose(loss, expected_loss))


@pytest.mark.parametrize("use_class_balancing", [True, False])
def test_focal_loss_cross_entropy_equivalence(use_class_balancing: bool) -> None:
    """
    Focal loss and cross-entropy loss should be equivalent to each other when the gamma parameter is set to zero.
    And this should also be independent from the class balancing term.
    """
    power = 1.0 if use_class_balancing else 0.0
    loss_fn_wout_focal_loss = CrossEntropyLoss(class_weight_power=power, focal_loss_gamma=None)
    loss_fn_w_focal_loss = CrossEntropyLoss(class_weight_power=power, focal_loss_gamma=0.0)

    class_indices = torch.randint(0, 5, torch.Size([1, 16, 16]))
    target = torch.nn.functional.one_hot(class_indices, num_classes=5).float().permute([0, 3, 1, 2])
    logits = torch.rand(torch.Size([1, 5, 16, 16]))

    loss1 = loss_fn_wout_focal_loss(logits, target)
    loss2 = loss_fn_w_focal_loss(logits, target)
    assert (torch.isclose(loss1, loss2))

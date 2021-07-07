#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any
from unittest import mock

import pytest
import torch

from InnerEye.ML.models.losses.soft_dice import SoftDiceLoss

# Set random seed
torch.random.manual_seed(1)

batch_size = 10
classes = 2
valid_spatial_sizes = [(3,), (3, 4), (3, 4, 5), (3, 4, 5, 6), (3, 4, 5, 6, 7)]


def random_output(spatial: Any) -> torch.Tensor:
    return torch.nn.functional.softmax(torch.rand(batch_size, classes, *spatial).float() * 100, dim=1)


valid_random_outputs = [random_output(spatial) for spatial in valid_spatial_sizes]
valid_random_targets = [torch.randint_like(output, low=0, high=2).float() for output in valid_random_outputs]

invalid_shapes = [(3,), (3, 4)]
invalid_shape_outputs = [torch.rand(*shape).float() for shape in invalid_shapes]  # type: ignore
invalid_shape_targets = [torch.randint_like(output, low=0, high=2).float() for output in invalid_shape_outputs]

dice_loss_f = SoftDiceLoss(eps=0, apply_softmax=False)


@pytest.mark.parametrize("output_target", list(zip(valid_random_outputs, valid_random_targets)))
def test_several_valid_spatial_sizes(output_target: Any) -> None:
    dice_loss_f(output=output_target[0], target=output_target[1])


@pytest.mark.parametrize("output", [None, list()])
@pytest.mark.parametrize("target", [None, list()])
def test_invalid_types(output: Any, target: Any) -> None:
    with pytest.raises(TypeError):
        dice_loss_f(output, target)


@pytest.mark.parametrize("output_target", list(zip(invalid_shape_outputs, invalid_shape_targets)))
def test_invalid_shapes(output_target: Any) -> None:
    with pytest.raises(ValueError):
        dice_loss_f(output_target[0], output_target[1])


def test_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        dice_loss_f(valid_random_outputs[0], valid_random_targets[1])


def test_zero_dice_loss() -> None:
    all_correct_output_target = torch.zeros(10, 2, 4)
    all_correct_output_target[..., 0, ...] = 1

    assert dice_loss_f(all_correct_output_target, all_correct_output_target).item() == 0


def test_one_dice_loss() -> None:
    all_wrong_output, all_wrong_target = torch.zeros(10, 2, 4), torch.zeros(10, 2, 4)
    all_wrong_output[..., 0, ...] = 1
    all_wrong_target[..., 1, ...] = 1

    assert dice_loss_f(all_wrong_output, all_wrong_target).item() == 1


def test_half_dice_loss() -> None:
    half_right_output, half_right_target = torch.zeros(10, 2, 4), torch.zeros(10, 2, 4)
    half_right_output[..., 0, 0:2] = 1
    half_right_output[..., 1, 2:4] = 1
    half_right_target[..., 0, 1:3] = 1
    half_right_target[..., 1, 0:4:2] = 1

    assert dice_loss_f(half_right_output, half_right_target).item() == 0.5


def test_dice_loss_regression() -> None:
    """
    Regression tests for specific values of the Dice loss for random input.
    """
    batch_size = 10
    classes = 3
    spatial = (3, 4, 5)
    total_size = (batch_size, classes, *spatial)
    dice = SoftDiceLoss()
    expected_results = [0.613937258720398, 0.6056110262870789, 0.6131367683410645]
    for index, seed in enumerate([1, 2, 3]):
        torch.random.manual_seed(seed)
        random_output = torch.rand(*total_size).float() * 100
        random_targets = torch.randint_like(random_output, low=0, high=2).float()
        result = dice.forward_minibatch(random_output, random_targets)
        assert result.item() == expected_results[index]


def test_dice_loss_2gpus() -> None:
    """
    Regression tests for specific values of the Dice loss for random input, simulating that two GPUs are present.
    """
    batch_size = 10
    classes = 3
    spatial = (3, 4, 5)
    total_size = (batch_size, classes, *spatial)
    dice = SoftDiceLoss()
    expected_results = [0.613937258720398, 0.6056110262870789, 0.6131367683410645]

    with mock.patch()
    for index, seed in enumerate([1, 2, 3]):
        torch.random.manual_seed(seed)
        random_output = torch.rand(*total_size).float() * 100
        random_targets = torch.randint_like(random_output, low=0, high=2).float()
        result = dice.forward_minibatch(random_output, random_targets)
        assert result.item() == expected_results[index]


def test_dice_loss_with_power() -> None:
    """
    Regression tests for Dice loss when class weight power is used.
    """
    batch_size = 10
    classes = 3
    spatial = (3, 4, 5)
    total_size = (batch_size, classes, *spatial)
    dice = SoftDiceLoss(class_weight_power=0.5)
    expected_results = [0.6104415655136108, 0.6056110262870789, 0.6131367683410645]
    for index, seed in enumerate([1, 2, 3]):
        torch.random.manual_seed(seed)
        random_output = torch.rand(*total_size).float() * 100
        random_targets = torch.randint_like(random_output, low=0, high=2).float()
        result = dice.forward_minibatch(random_output, random_targets)
        assert result.item() == expected_results[index]

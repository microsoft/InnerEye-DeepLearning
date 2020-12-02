#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Tuple

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from InnerEye.ML.deep_learning_config import TemperatureScalingConfig
from InnerEye.ML.models.losses.ece import ECELoss
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.temperature_scaling import ModelWithTemperature
from Tests.ML.test_config_helpers import IdentityModel


def test_set_temperature() -> None:
    """
    Test to make sure a temperature scale parameter that optimizes calibration of the logits is learnt
    """
    ml_util.set_random_seed(0)
    ece_loss_fn = ECELoss(activation=torch.nn.functional.sigmoid)
    loss_fn = BCEWithLogitsLoss()
    model: ModelWithTemperature = ModelWithTemperature(
        model=IdentityModel(),
        temperature_scaling_config=TemperatureScalingConfig(lr=0.1, max_iter=10)
    )
    # Temperature should not be learnt during model training
    assert model.temperature.requires_grad == False

    def criterion_fn(_logits: torch.Tensor, _labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return loss_fn(_logits, _labels), ece_loss_fn(_logits, _labels)

    labels = torch.rand(size=(5, 1))
    logits = torch.ones_like(labels)

    before_loss, before_ece = criterion_fn(logits, labels)
    optimal_temperature = model.set_temperature(logits, labels, criterion_fn, use_gpu=False)
    after_loss, after_ece = criterion_fn(model(logits), labels)
    assert after_loss.item() < before_loss.item()
    assert after_ece.item() < before_ece.item()
    assert np.isclose(optimal_temperature, 1.44, rtol=0.1)
    assert model.temperature.requires_grad == False


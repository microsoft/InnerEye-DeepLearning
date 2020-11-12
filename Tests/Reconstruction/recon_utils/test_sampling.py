#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.Reconstruction.recon_utils import sampling
import torch
from pytest import approx, raises

def test_random_undersampling_has_correct_acceleration_factor() -> None:
    sp = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2.0, centerlines=0, use_random_sampling=True, seed=42)
    assert torch.sum(sp) == approx(sp.numel() / 2.0)
    sp = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=1.5, centerlines=0, use_random_sampling=True, seed=42)
    assert torch.sum(sp) == approx(sp.numel() / 1.5, abs=.5)

def test_random_sampling_takes_seed() -> None:
    sp1 = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2.0, centerlines=0, use_random_sampling=True, seed=42)
    sp2 = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2.0, centerlines=0, use_random_sampling=True, seed=42)
    sp3 = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2.0, centerlines=0, use_random_sampling=True, seed=100)
    assert torch.sum(sp1 - sp2) == approx(0.0)
    assert torch.sum(torch.abs(sp1 - sp3)) > 1.0

def test_center_is_fully_sampled() -> None:
    sp = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2.0, centerlines=16, use_random_sampling=True, seed=42)
    assert torch.mean(sp[8:24, :]) == approx(1.0, abs=0.001)

def test_regular_undersampling_raises_error_when_matrix_not_multiple_of_acceleration_factor() -> None:
    with raises(Exception):
        _ = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=3, centerlines=0, use_random_sampling=False)

def test_regular_undersampling_is_regular() -> None:
    sp = sampling.generate_cartesian_sampling_pattern((32, 1), acceleration_factor=2, centerlines=0, use_random_sampling=False)
    sp0 = sp[0::2, :]
    sp1 = sp[1::2, :]
    assert torch.sum(sp0) == sp0.numel()
    assert torch.sum(sp1) == approx(0.0, abs=1e-5)

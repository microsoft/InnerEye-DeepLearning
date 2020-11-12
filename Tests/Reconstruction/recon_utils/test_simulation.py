#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import h5py
import math
import torch
from pytest import approx

from InnerEye.Reconstruction.recon_utils import simulation
from Tests.fixed_paths_for_tests import full_reconstruction_test_data_path

sim_data_path = full_reconstruction_test_data_path() / 'simulation.h5'


def test_shepp_logan_definitions_has_correct_numer_of_ellipses() -> None:
    assert len(simulation.shepp_logan_definition()) == 10
    assert len(simulation.modified_shepp_logan_definition()) == 10


def test_phantom_generation_produces_correct_shapes() -> None:
    mod_shepp_logan = simulation.phantom(16)
    std_shepp_logan = simulation.phantom(32, phantom_type=simulation.shepp_logan_definition)

    assert mod_shepp_logan.shape == (16, 16)
    assert torch.max(mod_shepp_logan) == 1.0
    assert std_shepp_logan.shape == (32, 32)
    assert torch.max(std_shepp_logan) == 2.0


def test_random_phantom_has_expected_signal_properties() -> None:
    number_of_ellipses = [1, 2, 4, 10, 50]
    for ne in number_of_ellipses:
        phan = simulation.phantom(matrix_size=32,
                                  phantom_type=lambda: simulation.random_phantom_definition(ellipses=ne))
        assert torch.max(phan) <= ne
        assert torch.unique(phan).numel() <= ne ** 2 + 1


def test_birdcage_sensitivities_have_expected_dimensions_and_statistics() -> None:
    normalized = simulation.generate_birdcage_sensitivities(matrix_size=32, number_of_coils=4)
    assert normalized.shape == (4, 32, 32)
    assert torch.sum(normalized.angle().abs()) > 0.1  # There should be some phase
    rss = torch.sqrt(torch.sum(torch.abs(normalized * normalized.conj()), dim=0))
    assert torch.mean(rss[rss > 0.01]) == approx(1.0, abs=1e-2)  # Test normalization
    notnormalized = simulation.generate_birdcage_sensitivities(matrix_size=32, number_of_coils=4, normalize=False)
    rss = torch.sqrt(torch.sum(torch.abs(notnormalized * notnormalized.conj()), dim=0))
    assert torch.mean(rss[rss > 0.01]) != approx(1.0, abs=1e-2)  # Test normalization disabled


def test_simulated_intensities_match_expected_signal_values() -> None:
    gen_phan = simulation.phantom(64)
    gen_coils = simulation.generate_birdcage_sensitivities(matrix_size=64, number_of_coils=4)
    gen_coil_img = gen_phan * gen_coils
    with h5py.File(sim_data_path, 'r') as f:
        phan = torch.from_numpy(f['phantom'][...])
        csm = torch.from_numpy(f['coil_sensitivity_maps'][...])
        coil_img = torch.from_numpy(f['coil_images'][...])
    assert torch.sum(csm - gen_coils) == approx(0.0, abs=0.001)
    assert torch.sum(phan - gen_phan) == approx(0.0, abs=0.001)
    assert torch.sum(gen_coil_img - coil_img) == approx(0.0, abs=0.001)

def test_phase_roll_generation_rotation() -> None:
    phase0 = simulation.generate_phase_roll(matrix_size=33, roll=math.pi/2)
    phase90 = simulation.generate_phase_roll(matrix_size=33, rotation=math.pi/2, roll=math.pi/2)

    # phase0 should have no phase roll in the y dimension, should have pi in the x direction
    assert torch.sum(torch.angle(phase0[0, :])-torch.angle(phase0[1, :])) == approx(0.0)
    assert torch.abs(torch.angle(phase0[0, 0])-torch.angle(phase0[0, 32])) == approx(math.pi)

    # phase9 should have no phase roll in the x dimension, should have pi in the y direction
    assert torch.sum(torch.angle(phase90[:, 0])-torch.angle(phase90[:, 1])) == approx(0.0, abs=1e-5)
    assert torch.abs(torch.angle(phase90[0, 0])-torch.angle(phase90[32, 0])) == approx(math.pi, abs=1e-5)


def test_phase_roll_center() -> None:
    phase_center_1 = simulation.generate_phase_roll(matrix_size=33, roll=math.pi/2, center=(1.0, 1.0))
    assert torch.angle(phase_center_1[32, 32]) == approx(0, abs=1e-7)

def test_random_phase_roll() -> None:
    phase = simulation.generate_random_phase_roll(matrix_size=32)
    assert torch.sum(torch.abs(torch.angle(phase))) > math.pi

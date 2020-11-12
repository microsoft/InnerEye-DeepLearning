#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path

import h5py
import torch
from pytest import approx

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.dataset import fastmri_dataset
from InnerEye.Reconstruction.recon_utils import fastmri_simulator, fft


def test_fastmri_simulation_generates_expected_files_and_slices(test_output_dirs: OutputFolderForTests) -> None:
    config = fastmri_simulator.FastMriSimulationConfig()
    config.output = test_output_dirs.root_dir
    config.name = "fastmri-test-simulation"
    config.matrix = 32
    config.slices = 4
    config.datasets = 2
    config.coils = 3

    fastmri_simulator.generate_fastmri_datasets(config)

    generated_files = list((Path(test_output_dirs.root_dir) / config.name).glob("*.h5"))
    generated_files.sort()
    assert len(generated_files) == config.datasets

    # Check that file names follow pattern
    assert generated_files[0].name == f"{fastmri_dataset.FILE_PREFIX_NAME}{0:0>6d}.h5"
    assert generated_files[1].name == f"{fastmri_dataset.FILE_PREFIX_NAME}{1:0>6d}.h5"

    with h5py.File(generated_files[0], 'r') as f:
        kspace = f[fastmri_dataset.KSPACE_NAME][...]
        assert kspace.shape == (config.slices,
                                config.coils,
                                config.matrix,
                                config.matrix)

        reconstruction_rss = f[fastmri_dataset.RECONSTRUCTION_NAME][...]
        assert reconstruction_rss.shape == (config.slices,
                                            config.matrix,
                                            config.matrix)

        coil_sensitivities = f[fastmri_dataset.COIL_SENSIVITY_NAME][...]
        assert coil_sensitivities.shape == (config.slices,
                                            config.coils,
                                            config.matrix,
                                            config.matrix)

        # See if the reconstruction makes sense
        kspace = torch.tensor(kspace)
        reconstruction_rss = torch.tensor(reconstruction_rss)
        recon = fft.ifft(kspace, dim=(-2, -1))
        recon = torch.sqrt(torch.sum(recon * recon.conj(), dim=1)).squeeze()
        assert torch.sum(recon - reconstruction_rss) == approx(0.0, abs=1e-3)

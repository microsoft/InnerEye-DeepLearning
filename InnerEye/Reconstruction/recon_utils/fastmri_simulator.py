#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from datetime import datetime
import math
from pathlib import Path
import random

import h5py
import param
import torch

from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.dataset import fastmri_dataset
from InnerEye.Reconstruction.recon_utils import simulation
from InnerEye.Reconstruction.recon_utils import fft

class FastMriSimulationConfig(GenericConfig):
    """
    Stores all information that is need to generate a simulated fastMRI (fastmri.org) dataset.
    """
    datasets: int = param.Integer(default=1, doc="Number of datasets (files) to generate")
    slices: int = param.Integer(default=1, doc="Number of slices per datasets")
    coils: int = param.Integer(default=2, doc="Number of simulated received coils")
    matrix: int = param.Integer(default=32, doc="Matrix size")
    name: str = param.String(default='fastmrisim-' + datetime.now().strftime('%Y%m%d-%H%M%S'), doc="Name of dataset")
    output: Path = param.ClassSelector(class_=Path, default=Path.cwd(), doc="Output folder")

def generate_random_fastmri_dataset(filename: Path, matrix_size: int = 256, coils: int = 2, slices: int = 1) -> None:

    """
    Generates a random simulated MRI dataset in the fastMRI (fastmri.org) form and stores it in and HDF5 file.

    :param filename: Path to file location where dataset should be written. Cannot exist already.
    :param matrix_size: Size of imaging matrix
    :param coils: Number of simulated receiver coils
    :param slices: Number of simulated slices
    """

    if filename.exists():
        raise ValueError("Output file for simulation already exists")

    kspace = torch.zeros(size=(slices, coils, matrix_size, matrix_size), dtype=torch.cfloat)
    reconstruction_rss = torch.zeros(size=(slices, matrix_size, matrix_size), dtype=torch.float)
    coil_sensitivities = torch.zeros(size=(slices, coils, matrix_size, matrix_size), dtype=torch.cfloat)

    for s in range(slices):
        phan = simulation.phantom(matrix_size=matrix_size, phantom_type=simulation.random_phantom_definition)
        phan = phan * simulation.generate_random_phase_roll(matrix_size=matrix_size)
        csm = simulation.generate_birdcage_sensitivities(matrix_size=matrix_size, number_of_coils=coils, rotation=2*random.random()*math.pi)
        coil_img = phan*csm
        recon = torch.sqrt(torch.sum(coil_img*coil_img.conj(), dim=0))
        k = fft.fft(coil_img, dim=(-2, -1))
        kspace[s, ...] = k
        reconstruction_rss[s, ...] = recon
        coil_sensitivities[s, ...] = csm

    with h5py.File(filename, 'w') as f:
        f.create_dataset(fastmri_dataset.KSPACE_NAME, data=kspace.numpy())
        f.create_dataset(fastmri_dataset.RECONSTRUCTION_NAME, data=reconstruction_rss.numpy())
        f.create_dataset(fastmri_dataset.COIL_SENSIVITY_NAME, data=coil_sensitivities.numpy())

def generate_fastmri_datasets(config: FastMriSimulationConfig) -> None:
    """
    Generates a number of simulated MRI datasets that match the fastMRI format.

    :param config: FastMriSimulationConfiguration
    """

    datasetpath = config.output/config.name
    if datasetpath.exists():
        raise ValueError("Dataset path: " + str(datasetpath) + " already exists. Choose another name or move existing dataset.")
    else:
        datasetpath.mkdir(parents=True)

    for d in range(config.datasets):
        outfile = datasetpath/f"{fastmri_dataset.FILE_PREFIX_NAME}{d:0>6d}.h5"
        generate_random_fastmri_dataset(filename=outfile,
                                        matrix_size=config.matrix,
                                        coils=config.coils,
                                        slices=config.slices)

def main() -> None:
    simulation_config = FastMriSimulationConfig.parse_args()
    generate_fastmri_datasets(simulation_config)


if __name__ == '__main__':
    main()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List

import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from pytest import approx
import torch

from InnerEye.Reconstruction.recon_utils import fft

def test_permute_util() -> None:
    assert fft.get_fft_permute_dims(5, (0, 1)) == ((2, 3, 4, 0, 1), (3, 4, 0, 1, 2))
    assert fft.get_fft_permute_dims(5, (-2, -1)) == ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4))

def test_fft_noise_scaling() -> None:
    # Avoid making this tensor too small since, statistics (below) become less reliable
    noise = torch.randn(256, 256, dtype=torch.cfloat)
    assert torch.std(noise) == approx(1.0, abs=1e-2)

    # Noise std in k-space should also be 1.0 
    knoise = fft.fft(noise)
    assert torch.std(knoise) == approx(1.0, abs=1e-2)

    # Noise after recon should also be 1.0 
    rnoise = fft.fft(knoise)
    assert torch.std(rnoise) == approx(1.0, abs=1e-2)

def test_fft() -> None:
    np_object = np.zeros((4, 4, 32, 32), dtype=np.dtype('complex64'))

    # We will make a square in each image
    np_object[:, :, 8:24, 8:24] = 1 + 1j*1
    torch_test_object = torch.from_numpy(np_object)

    # Let's FFT the last dimension:
    no1 = transform_image_to_kspace(np_object, dim=[3])
    to1 = fft.fft(torch_test_object, dim=3).numpy()
    assert np.mean(np.sqrt(np.abs(no1-to1)**2)) < 1e-7

    # Let's FFT the second to last dimension:
    no1 = transform_image_to_kspace(np_object, dim=[-2])
    to1 = fft.fft(torch_test_object, dim=-2).numpy()
    assert np.mean(np.sqrt(np.abs(no1-to1)**2)) < 1e-7

    # Let's FFT the last two dimensions
    no1 = transform_image_to_kspace(np_object, dim=[-1, -2])
    to1 = fft.fft(torch_test_object, dim=(-1, -2)).numpy()
    assert np.mean(np.sqrt(np.abs(no1-to1)**2)) < 1e-7

def test_phase_roll() -> None:
    int_phase_roll_test(matrix_size=32)
    int_phase_roll_test(matrix_size=33)

# Some internal reference functions for FFTs based on numpy 
def transform_kspace_to_image(k: np.ndarray, dim: List[int] = None) -> np.ndarray:
    if not dim:
        dim = [x for x in range(k.ndim)]

    img = fftshift(ifftn(ifftshift(k, axes=dim), axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img

def transform_image_to_kspace(img: np.ndarray, dim: List[int] = None) -> np.ndarray:
    if not dim:
        dim = [x for x in range(img.ndim)]

    k = fftshift(fftn(ifftshift(img, axes=dim), axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k

def int_phase_roll_test(matrix_size: int = 256) -> None:
    np_object = np.zeros((matrix_size, matrix_size), dtype=np.dtype('complex64'))

    # No phase in this object
    np_object[(matrix_size//2-matrix_size//4):(matrix_size//2+matrix_size//4), (matrix_size//2-matrix_size//4):(matrix_size//2+matrix_size//4)] = 1
    torch_test_object = torch.from_numpy(np_object)

    kspace = fft.fft(torch_test_object)
    recon = fft.ifft(kspace)

    # After FFT to k-space and back again. there should be no phase
    assert torch.max((recon.abs() > 0.001) * recon.angle()) < 1e-5

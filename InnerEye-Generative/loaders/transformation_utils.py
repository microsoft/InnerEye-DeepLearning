#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import SimpleITK as sitk

class ToNumpy(object):
    def __call__(self, sample):
        return np.array(sample)

def resample_image(image, out_spacing=(1.0, 1.0, 1.0), out_size=None, is_label=False, pad_value=0):
    """Resamples an image to given element spacing and output size."""

    original_spacing = np.array(image.GetSpacing())
    original_size = np.array(image.GetSize())

    if out_size is None:
        out_size = np.round(np.array(original_size * original_spacing / np.array(out_spacing))).astype(int)
    else:
        out_size = np.array(out_size)

    original_direction = np.array(image.GetDirection()).reshape(len(original_spacing), -1)
    original_center = (np.array(original_size, dtype=float) - 1.0) / 2.0 * original_spacing
    out_center = (np.array(out_size, dtype=float) - 1.0) / 2.0 * np.array(out_spacing)

    original_center = np.matmul(original_direction, original_center)
    out_center = np.matmul(original_direction, out_center)
    out_origin = np.array(image.GetOrigin()) + (original_center - out_center)

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size.tolist())
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(out_origin.tolist())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(pad_value)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image)

def zero_mean_unit_var(image, mask):
    """Normalizes an image to zero mean and unit variance."""

    img_array = sitk.GetArrayFromImage(image)
    img_array = img_array.astype(np.float32)

    msk_array = sitk.GetArrayFromImage(mask)

    mean = np.mean(img_array[msk_array > 0])
    std = np.std(img_array[msk_array > 0])

    if std > 0:
        img_array = (img_array - mean) / std
        img_array[msk_array == 0] = 0

    image_normalised = sitk.GetImageFromArray(img_array)
    image_normalised.CopyInformation(image)

    return image_normalised

def quantile_normalisation(img_array, q1: float = 0.02, q2: float = 0.98):
    """Normalizes an image mapping the 2% quantile to zero and 98% quantile to 1."""

    qs = np.quantile(img_array[img_array != 0], (q1, q2))

    assert qs[0] != qs[1]
    # solve aq1+b=0, aq2+b=1
    a = 1 / (qs[1] - qs[0])
    b = - a * qs[0]
    image_normalised = np.zeros_like(img_array).astype(float)
    image_normalised[img_array != 0] = img_array[img_array != 0] * a + b
    return image_normalised

def gamma_transform(img, gamma: float):
    mask = img == 0
    max = img.max()
    min = img.min()
    img = (img - min) / (max - min)
    img = np.power(img, gamma)
    img = img * (min + max) + min
    img[mask] = 0
    return img

def gamma_transform_mod(img, gamma):
    img[img != 0] = np.power(img, gamma)[img != 0]
    return img 

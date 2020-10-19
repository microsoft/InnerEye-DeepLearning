#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.config import SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.dataset.sample import PatientMetadata, Sample
from InnerEye.ML.plotting import resize_and_save, scan_and_transparent_overlay
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.image_util import get_unit_image_header
from InnerEye.ML.visualizers.patch_sampling import visualize_patch_sampling
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.parametrize("labels_to_boundary", [True, False])
def test_visualize_patch_sampling(test_output_dirs: TestOutputDirectories,
                                  labels_to_boundary: bool) -> None:
    """
    Tests if patch sampling and producing diagnostic images works as expected.
    :param test_output_dirs:
    :param labels_to_boundary: If true, the ground truth labels are placed close to the image boundary, so that
    crops have to be adjusted inwards. If false, ground truth labels are all far from the image boundaries.
    """
    np.random.seed(0)
    shape = (10, 30, 30)
    foreground_classes = ["fg"]
    class_weights = equally_weighted_classes(foreground_classes)
    config = SegmentationModelBase(should_validate=False,
                                   crop_size=(2, 10, 10),
                                   class_weights=class_weights)
    image = np.random.rand(1, *shape).astype(np.float32) * 1000
    mask = np.ones(shape)
    labels = np.zeros((len(class_weights),) + shape)
    if labels_to_boundary:
        # Generate foreground labels in such a way that a patch centered around a foreground pixel would
        # reach outside of the image.
        labels[1, 4:8, 3:27, 3:27] = 1
    else:
        labels[1, 4:8, 15:18, 15:18] = 1
    labels[0] = 1 - labels[1]
    output_folder = Path(test_output_dirs.root_dir)
    image_header = get_unit_image_header()
    io_util.store_as_ubyte_nifti(labels[1], image_header, str(output_folder / "labels.nii.gz"))
    sample = Sample(image=image,
                    mask=mask,
                    labels=labels,
                    metadata=PatientMetadata(patient_id=123,
                                             image_header=image_header))
    visualize_patch_sampling(sample, config, output_folder=output_folder)
    f1 = output_folder / "123_ct.nii.gz"
    assert f1.is_file()
    f2 = output_folder / "123_sampled_patches.nii.gz"
    assert f2.is_file()

    expected_folder = full_ml_test_data_path("patch_sampling")
    expected = "sampled_to_boundary.nii.gz" if labels_to_boundary else "sampled_center.nii.gz"
    expected_image = io_util.load_nifti_image(expected_folder / expected)
    actual_image = io_util.load_nifti_image(f2)
    np.allclose(expected_image.image, actual_image.image)


@pytest.mark.skipif(is_windows(), reason="Plotting output is not consistent across platforms.")
@pytest.mark.parametrize("dimension", [0, 1, 2])
def test_plot_overlay(test_output_dirs: TestOutputDirectories,
                      dimension: int) -> None:
    np.random.seed(0)
    shape = (10, 30, 30)
    image = np.random.rand(*shape).astype(np.float32) * 1000
    mask = np.zeros(shape).flatten()
    for i in range(len(mask)):
        mask[i] = i
    mask = mask.reshape(shape)
    plt.figure()
    scan_and_transparent_overlay(image, mask, dimension, shape[dimension] // 2)
    file = Path(test_output_dirs.root_dir) / "plot.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("patch_sampling") / f"overlay_{dimension}.png"
    # To update the stored results:
    # expected.write_bytes(file.read_bytes())
    assert file.read_bytes() == expected.read_bytes()

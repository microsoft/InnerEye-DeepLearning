#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.dataset.sample import PatientMetadata, Sample
from InnerEye.ML.plotting import resize_and_save, scan_with_transparent_overlay
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.image_util import get_unit_image_header
from InnerEye.ML.utils.io_util import load_nifti_image
from InnerEye.ML.utils.ml_util import set_random_seed
from InnerEye.ML.visualizers.patch_sampling import visualize_random_crops
from Tests.ML.util import assert_binary_files_match, assert_file_exists, is_running_on_azure
from Tests.fixed_paths_for_tests import full_ml_test_data_path


@pytest.mark.skipif(is_windows(), reason="Plotting output is not consistent across platforms.")
@pytest.mark.parametrize("labels_to_boundary", [True, False])
def test_visualize_patch_sampling(test_output_dirs: OutputFolderForTests,
                                  labels_to_boundary: bool) -> None:
    """
    Tests if patch sampling and producing diagnostic images works as expected.
    :param test_output_dirs:
    :param labels_to_boundary: If true, the ground truth labels are placed close to the image boundary, so that
    crops have to be adjusted inwards. If false, ground truth labels are all far from the image boundaries.
    """
    set_random_seed(0)
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
    sample = Sample(image=image,
                    mask=mask,
                    labels=labels,
                    metadata=PatientMetadata(patient_id='123',
                                             image_header=image_header))
    expected_folder = full_ml_test_data_path("patch_sampling")
    heatmap = visualize_random_crops(sample, config, output_folder=output_folder)
    expected_heatmap = expected_folder / ("sampled_to_boundary.npy" if labels_to_boundary else "sampled_center.npy")
    # To update the stored results, uncomment this line:
    # np.save(str(expected_heatmap), heatmap)
    assert np.allclose(heatmap, np.load(str(expected_heatmap))), "Patch sampling created a different heatmap."
    f1 = output_folder / "123_ct.nii.gz"
    assert_file_exists(f1)
    f2 = output_folder / "123_sampled_patches.nii.gz"
    assert_file_exists(f2)
    thumbnails = [
        "123_sampled_patches_dim0.png",
        "123_sampled_patches_dim1.png",
        "123_sampled_patches_dim2.png",
    ]
    for f in thumbnails:
        assert_file_exists(output_folder / f)

    expected = expected_folder / ("sampled_to_boundary.nii.gz" if labels_to_boundary else "sampled_center.nii.gz")
    # To update test results:
    # shutil.copy(str(f2), str(expected))
    expected_image = io_util.load_nifti_image(expected)
    actual_image = io_util.load_nifti_image(f2)
    np.allclose(expected_image.image, actual_image.image)
    if labels_to_boundary:
        for f in thumbnails:
            # Uncomment this line to update test results
            # (expected_folder / f).write_bytes((output_folder / f).read_bytes())
            if not is_running_on_azure():
                # When running on the Azure build agents, it appears that the bounding box of the images
                # is slightly different than on local runs, even with equal dpi settings.
                # Not able to figure out how to make the run results consistent, hence disable in cloud runs.
                assert_binary_files_match(output_folder / f, expected_folder / f)


@pytest.mark.skipif(is_windows(), reason="Plotting output is not consistent across platforms.")
def test_visualize_patch_sampling_2d(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests if patch sampling works for 2D images.
    :param test_output_dirs:
    """
    set_random_seed(0)
    shape = (1, 20, 30)
    foreground_classes = ["fg"]
    class_weights = equally_weighted_classes(foreground_classes)
    config = SegmentationModelBase(should_validate=False,
                                   crop_size=(1, 5, 10),
                                   class_weights=class_weights)
    image = np.random.rand(1, *shape).astype(np.float32) * 1000
    mask = np.ones(shape)
    labels = np.zeros((len(class_weights),) + shape)
    labels[1, 0, 8:12, 5:25] = 1
    labels[0] = 1 - labels[1]
    output_folder = Path(test_output_dirs.root_dir)
    image_header = None
    sample = Sample(image=image,
                    mask=mask,
                    labels=labels,
                    metadata=PatientMetadata(patient_id='123',
                                             image_header=image_header))
    heatmap = visualize_random_crops(sample, config, output_folder=output_folder)
    expected_folder = full_ml_test_data_path("patch_sampling")
    expected_heatmap = expected_folder / "sampling_2d.npy"
    # To update the stored results, uncomment this line:
    # np.save(str(expected_heatmap), heatmap)
    assert np.allclose(heatmap, np.load(str(expected_heatmap))), "Patch sampling created a different heatmap."
    assert len(list(output_folder.rglob("*.nii.gz"))) == 0
    assert len(list(output_folder.rglob("*.png"))) == 1
    actual_file = output_folder / "123_sampled_patches.png"
    assert_file_exists(actual_file)
    expected = expected_folder / "sampling_2d.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(actual_file.read_bytes())
    if not is_running_on_azure():
        # When running on the Azure build agents, it appears that the bounding box of the images
        # is slightly different than on local runs, even with equal dpi settings.
        # It says: Image sizes don't match: actual (685, 469), expected (618, 424)
        # Not able to figure out how to make the run results consistent, hence disable in cloud runs.
        assert_binary_files_match(actual_file, expected)


@pytest.mark.skipif(is_windows(), reason="Plotting output is not consistent across platforms.")
@pytest.mark.parametrize("dimension", [0, 1, 2])
def test_plot_overlay(test_output_dirs: OutputFolderForTests,
                      dimension: int) -> None:
    set_random_seed(0)
    shape = (10, 30, 30)
    image = np.random.rand(*shape).astype(np.float32) * 1000
    mask = np.zeros(shape).flatten()
    for i in range(len(mask)):
        mask[i] = i
    mask = mask.reshape(shape)
    plt.figure()
    scan_with_transparent_overlay(image, mask, dimension, shape[dimension] // 2, spacing=(1.0, 1.0, 1.0))
    file = Path(test_output_dirs.root_dir) / "plot.png"
    resize_and_save(5, 5, file)
    assert file.exists()
    expected = full_ml_test_data_path("patch_sampling") / f"overlay_{dimension}.png"
    # To update the stored results, uncomment this line:
    # expected.write_bytes(file.read_bytes())
    assert_binary_files_match(file, expected)


@pytest.mark.skipif(is_windows(), reason="Plotting output is not consistent across platforms.")
def test_show_non_square_images(test_output_dirs: OutputFolderForTests) -> None:
    input_file = full_ml_test_data_path("patch_sampling") / "scan_small.nii.gz"
    input = load_nifti_image(input_file)
    image = input.image
    shape = image.shape
    mask = np.zeros_like(image)
    mask[shape[0] // 2, shape[1] // 2, shape[2] // 2] = 1
    for dim in range(3):
        scan_with_transparent_overlay(image, mask, dim, shape[dim] // 2, spacing=input.header.spacing)
        actual_file = Path(test_output_dirs.root_dir) / f"dim_{dim}.png"
        resize_and_save(5, 5, actual_file)
        expected = full_ml_test_data_path("patch_sampling") / f"overlay_with_aspect_dim{dim}.png"
        # To update the stored results, uncomment this line:
        # expected.write_bytes(actual_file.read_bytes())
        assert_binary_files_match(actual_file, expected)

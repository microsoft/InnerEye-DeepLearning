#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import itertools
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pytest

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML import plotting
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import Sample
from InnerEye.ML.photometric_normalization import PhotometricNormalization, PhotometricNormalizationMethod
from Tests.ML.util import DummyPatientMetadata
from Tests.fixed_paths_for_tests import full_ml_test_data_path


def file_as_bytes(name: Union[str, Path]) -> bytes:
    if isinstance(name, str):
        name = Path(name)
    return name.read_bytes()


@pytest.mark.parametrize("num_classes", [3, 15])
def test_plot_dice_per_epoch(test_output_dirs: OutputFolderForTests, num_classes: int) -> None:
    metrics: Dict[str, Any] = {}
    epoch = [1, 2, 3]
    for i in range(num_classes):
        metric_name = "Val_Dice/Class{}".format(i)
        loss = [i / num_classes * j / 3 for j in epoch]
        metrics[metric_name] = {"epoch": epoch, "loss": loss}
    metrics["baz"] = [17]
    series_count = plotting.plot_val_dice_per_epoch(metrics)
    file_name = test_output_dirs.root_dir / f"dice_per_epoch_{num_classes}classes.png"
    plotting.add_legend(series_count)
    plotting.resize_and_save(5, 4, file_name)
    assert file_name.is_file()
    # Try writing the same figure again, to see what the file overwrite behaviour is.
    # In actual training runs, the file will be overwritten repeatedly.
    plotting.resize_and_save(5, 4, file_name)
    # It would be good to compare the resulting plot with an expected plot,
    # but the results on the local box are always different from those in Azure builds.
    # expected = fixed_paths.full_test_data_path(file_name)
    # assert file_as_bytes(file_name) == file_as_bytes(expected)


def test_plot_image_and_contour(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test plotting of an image with an overlaid contour.
    """
    size = (3, 3)
    image = np.zeros(size)
    image[0, 0] = -1
    image[2, 2] = 1
    labels = np.zeros(size)
    labels[1, 1] = 1
    file_name = "image_and_contour.png"
    plot_file = test_output_dirs.root_dir / file_name
    plotting.plot_image_and_label_contour(image, labels, contour_arguments={'colors': 'r'}, plot_file_name=plot_file)
    assert plot_file.exists()
    expected = full_ml_test_data_path(file_name)
    # To update the stored results, uncomment this line:
    # expected.write_bytes(plot_file.read_bytes())
    assert file_as_bytes(plot_file) == file_as_bytes(expected)


def test_plot_image_and_contour_scaled(test_output_dirs: OutputFolderForTests) -> None:
    """
    When providing an additional scaling that is a lot larger than the image range,
    the output should be mostly grey.
    """
    size = (3, 3)
    image = np.zeros(size)
    image[0, 0] = -1
    image[2, 2] = 1
    labels = np.zeros(size)
    labels[1, 1] = 1
    file_name = "image_scaled_and_contour.png"
    plot_file = test_output_dirs.root_dir / file_name
    plotting.plot_image_and_label_contour(image, labels, contour_arguments={'colors': 'b'},
                                          image_range=(-5, 5), plot_file_name=plot_file)
    assert plot_file.exists()
    expected = full_ml_test_data_path(file_name)
    # To update the stored results, uncomment this line:
    # expected.write_bytes(plot_file.read_bytes())
    assert file_as_bytes(plot_file) == file_as_bytes(expected)


def test_plot_image_and_multiple_contours(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test plotting of an image with two overlaid contours.
    """
    size = (3, 3)
    image = np.zeros(size)
    image[0, 0] = -1
    image[2, 2] = 1
    labels1 = np.zeros(size)
    labels1[1, 1] = 1
    labels2 = np.zeros(size)
    labels2[0, 0] = 1
    file_name = "image_and_multiple_contours.png"
    plot_file = test_output_dirs.root_dir / file_name
    args1 = {'colors': 'r', 'linestyles': 'dashed'}
    args2 = {'colors': 'b'}
    plotting.plot_image_and_label_contour(image, [labels1, labels2],
                                          contour_arguments=[args1, args2],
                                          plot_file_name=plot_file)
    assert plot_file.exists()
    expected = full_ml_test_data_path(file_name)
    # To update the stored results, uncomment this line:
    # expected.write_bytes(plot_file.read_bytes())
    assert file_as_bytes(plot_file) == file_as_bytes(expected)


def test_plot_contour_fails() -> None:
    with pytest.raises(ValueError) as ex:
        plotting.plot_image_and_label_contour(np.zeros((3, 3, 3)), np.zeros((2, 2)), plot_file_name=Path("x"))
    assert "should be a 2D array" in str(ex)
    with pytest.raises(ValueError) as ex:
        plotting.plot_image_and_label_contour(np.zeros((3, 3)), np.zeros((2, 2)), plot_file_name=Path("x"))
    assert "image and the contour data should have matching size" in str(ex)
    with pytest.raises(ValueError) as ex:
        plotting.plot_image_and_label_contour(np.zeros((3, 3)), [np.zeros((2, 2))], contour_arguments={'colors': 'r'},
                                              plot_file_name=Path("x"))
    assert "Combination of input arguments is not recognized" in str(ex)


def compare_files(actual: List[Path], expected: List[str]) -> None:
    assert len(actual) == len(expected)
    for (f, e) in zip(actual, expected):
        assert f.exists()
        full_expected = full_ml_test_data_path(e)
        assert full_expected.exists()
        assert str(f).endswith(e)
        # To update the stored results, uncomment this line:
        # full_expected.write_bytes(f.read_bytes())
        assert file_as_bytes(f) == file_as_bytes(full_expected)


@pytest.mark.skipif(common_util.is_windows(), reason="Rendering of the graph is slightly different on Linux")
def test_plot_normalization_result(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests plotting of before/after histograms in photometric normalization.
    :return:
    """
    size = (3, 3, 3)
    image = np.zeros((1,) + size)
    for i, (z, y, x) in enumerate(itertools.product(range(size[0]), range(size[1]), range(size[2]))):
        image[0, z, y, x] = i
    labels = np.zeros((2,) + size)
    labels[1, 1, 1, 1] = 1
    sample = Sample(
        image=image,
        labels=labels,
        mask=np.ones(size),
        metadata=DummyPatientMetadata
    )
    config = SegmentationModelBase(norm_method=PhotometricNormalizationMethod.CtWindow, window=4, level=13,
                                   should_validate=False)
    normalizer = PhotometricNormalization(config)
    folder = test_output_dirs.root_dir
    files = plotting.plot_normalization_result(sample, normalizer, folder)
    expected = ["042_slice_001.png", "042_slice_001_contour.png"]
    compare_files(files, expected)


def test_plot_contours_for_all_classes(test_output_dirs: OutputFolderForTests) -> None:
    size = (3, 3, 3)
    image = np.zeros((1,) + size)
    for i, (z, y, x) in enumerate(itertools.product(range(size[0]), range(size[1]), range(size[2]))):
        image[0, z, y, x] = i
    # Create a fake label array: For each class, there is exactly 1 pixel foreground, at the z slice that is
    # equal to the class index
    labels = np.zeros((3,) + size)
    labels[0, 0, 1, 1] = 1
    labels[1, 1, 1, 1] = 1
    labels[2, 2, 1, 1] = 1
    # Fake segmentation: Classifies all foreground pixels correctly...
    segmentation = np.zeros(size)
    segmentation[1, 1, 1] = 1
    segmentation[2, 1, 1] = 2
    # ...but has an extra foreground pixel in the largest z slice in either top left or bottom right corner:
    segmentation[1, 0, 0] = 1
    segmentation[2, 2, 2] = 2
    sample = Sample(
        image=image,
        labels=labels,
        mask=np.ones(size),
        metadata=DummyPatientMetadata
    )
    plots = plotting.plot_contours_for_all_classes(sample,
                                                   segmentation,
                                                   foreground_class_names=["class1", "class2"],
                                                   result_folder=test_output_dirs.root_dir,
                                                   result_prefix="prefix")
    expected = ["prefix042_class1_slice_001.png",
                "prefix042_class2_slice_002.png"]
    compare_files(plots, expected)
    with pytest.raises(ValueError) as err:
        plotting.plot_contours_for_all_classes(sample,
                                               segmentation,
                                               foreground_class_names=["background", "class1", "class2"],
                                               result_folder=test_output_dirs.root_dir,
                                               result_prefix="prefix")
    assert "3 classes" in str(err)
    assert "background" in str(err)

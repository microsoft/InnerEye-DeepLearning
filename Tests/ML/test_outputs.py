#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import DATASET_ID_FILE, GROUND_TRUTH_IDS_FILE, IMAGE_CHANNEL_IDS_FILE, \
    PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.model_testing import DEFAULT_RESULT_IMAGE_NAME, METRICS_AGGREGATES_FILE, store_inference_results, \
    store_run_information
from InnerEye.ML.pipelines.inference import InferencePipeline
from InnerEye.ML.plotting import resize_and_save
from InnerEye.ML.reports.segmentation_report import boxplot_per_structure
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.image_util import get_unit_image_header
from InnerEye.ML.utils.io_util import ImageHeader
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns
from InnerEye.ML.utils.metrics_util import MetricsPerPatientWriter
from InnerEye.ML.utils.transforms import LinearTransform, get_range_for_window_level
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import assert_file_contains_string, assert_nifti_content, assert_text_files_match
from Tests.fixed_paths_for_tests import full_ml_test_data_path

model_name = "Basic"
base_path = full_ml_test_data_path()

train_and_test_data_dir = full_ml_test_data_path("train_and_test_data")
ckpt_folder = train_and_test_data_dir / "expected" / model_name
default_image_name = "image.nii.gz"

dim_x = 1
dim_y = 2
dim_z = 3


def _create_config_with_folders(test_dirs: OutputFolderForTests) -> SegmentationModelBase:
    config = DummyModel()
    config.set_output_to(test_dirs.root_dir)
    return config


def to_unique_bytes(a: np.ndarray, input_range: Tuple[float, float]) -> Any:
    """Returns an array of unique ubytes after applying LinearTransform on the input array."""
    ubyte_range = (np.iinfo(np.ubyte).min, np.iinfo(np.ubyte).max)
    a = LinearTransform.transform(data=a, input_range=input_range, output_range=ubyte_range)
    return np.unique(a.astype(np.ubyte))


def test_store_inference_results(test_output_dirs: OutputFolderForTests) -> None:
    np.random.seed(0)
    num_classes = 2
    posterior = torch.nn.functional.softmax(
        torch.from_numpy(np.random.random_sample((num_classes, dim_z, dim_y, dim_x))), dim=0).numpy()
    segmentation = np.argmax(posterior, axis=0)
    assert segmentation.shape == (dim_z, dim_y, dim_x)

    posterior0 = to_unique_bytes(posterior[0], (0, 1))
    posterior1 = to_unique_bytes(posterior[1], (0, 1))
    spacing = (2.0, 2.0, 2.0)
    header = get_unit_image_header(spacing=spacing)
    inference_result = InferencePipeline.Result(
        epoch=1,
        patient_id=12,
        posteriors=posterior,
        segmentation=segmentation,
        voxel_spacing_mm=(1, 1, 1)
    )

    test_config = _create_config_with_folders(test_output_dirs)

    assert test_config.class_and_index_with_background() == {"background": 0, "region": 1}

    results_folder = test_output_dirs.root_dir
    store_inference_results(inference_result, test_config, Path(results_folder), header)

    assert_nifti_content(results_folder / "012" / "posterior_background.nii.gz",
                         segmentation.shape, header, list(posterior0), np.ubyte)

    assert_nifti_content(results_folder / "012" / "posterior_region.nii.gz",
                         segmentation.shape, header, list(posterior1), np.ubyte)

    assert_nifti_content(results_folder / "012" / "background.nii.gz",
                         segmentation.shape, header, list([0, 1]), np.ubyte)

    assert_nifti_content(results_folder / "012" / "region.nii.gz",
                         segmentation.shape, header, list([0, 1]), np.ubyte)

    assert_nifti_content(results_folder / "012" / DEFAULT_RESULT_IMAGE_NAME,
                         segmentation.shape, header, list(np.unique(segmentation)), np.ubyte)

    assert_nifti_content(results_folder / "012" / "uncertainty.nii.gz",
                         inference_result.uncertainty.shape, header, list([248, 249, 253, 254]), np.ubyte)


def test_metrics_file(test_output_dirs: OutputFolderForTests) -> None:
    """Test if metrics files with Dice scores are written as expected."""
    folder = test_output_dirs.make_sub_dir("test_metrics_file")

    def new_file(suffix: str) -> Path:
        file = folder / suffix
        if file.is_file():
            file.unlink()
        return file

    d = MetricsPerPatientWriter()
    p1 = "Patient1"
    p2 = "Patient2"
    p3 = "Patient3"
    liver = "liver"
    kidney = "kidney"
    # Ordering for test data: For "liver", patient 2 has the lowest score, sorting should move them first
    # For "kidney", patient 1 has the lowest score and should be first.
    d.add(p1, liver, 1.0, 1.0, 0.5)
    d.add(p1, liver, 0.4, 1.0, 0.4)
    d.add(p2, liver, 0.8, 1.0, 0.3)
    d.add(p2, kidney, 0.7, 1.0, 0.2)
    d.add(p3, kidney, 0.4, 1.0, 0.1)
    metrics_file = new_file("metrics_file.csv")
    d.to_csv(Path(metrics_file))
    # Sorting should be first by structure name alphabetically, then Dice with lowest scores first.
    assert_file_contains_string(metrics_file, "Patient,Structure,Dice,HausdorffDistance_mm,MeanDistance_mm\n"
                                              "Patient3,kidney,0.400,1.000,0.100\n"
                                              "Patient2,kidney,0.700,1.000,0.200\n"
                                              "Patient1,liver,0.400,1.000,0.400\n"
                                              "Patient2,liver,0.800,1.000,0.300\n"
                                              "Patient1,liver,1.000,1.000,0.500\n")
    aggregates_file = new_file(METRICS_AGGREGATES_FILE)
    d.save_aggregates_to_csv(Path(aggregates_file))
    # Sorting should be first by structure name alphabetically, then Dice with lowest scores first.
    assert_text_files_match(Path(aggregates_file),
                            full_ml_test_data_path() / METRICS_AGGREGATES_FILE)
    boxplot_per_structure(d.to_data_frame(),
                          column_name=MetricsFileColumns.DiceNumeric.value,
                          title="Dice score")
    boxplot1 = new_file("boxplot_2class.png")
    resize_and_save(5, 4, boxplot1)
    plt.clf()
    d.add(p1, "lung", 0.5, 2.0, 1.0)
    d.add(p1, "foo", 0.9, 2.0, 1.0)
    d.add(p1, "bar", 0.9, 2.0, 1.0)
    d.add(p1, "baz", 0.9, 2.0, 1.0)
    boxplot_per_structure(d.to_data_frame(),
                          column_name=MetricsFileColumns.DiceNumeric.value,
                          title="Dice score")
    boxplot2 = new_file("boxplot_6class.png")
    resize_and_save(5, 4, boxplot2)


def test_store_run_information(test_output_dirs: OutputFolderForTests) -> None:
    dataset_id = "placeholder_dataset_id"
    ground_truth_ids = ["id1", "id2"]
    channel_ids = ["channel1", "channel2"]
    results_folder = test_output_dirs.root_dir

    files = [results_folder / DATASET_ID_FILE,
             results_folder / GROUND_TRUTH_IDS_FILE,
             results_folder / IMAGE_CHANNEL_IDS_FILE]
    values = [[dataset_id], ground_truth_ids, channel_ids]
    store_run_information(results_folder, dataset_id, ground_truth_ids, channel_ids)
    for i, file in enumerate(files):
        assert file.exists()
        lines = [line.strip() for line in file.read_text().splitlines()]
        assert lines == values[i]


@pytest.mark.parametrize(["image_type", "scale", "input_range", "output_range"],
                         [(np.short, True, (0, 1), (0, 255)),
                          (np.short, False, (0, 1), (0, 1)),
                          (np.ubyte, False, None, None),
                          (np.short, False, (0, 1), None)])
def test_store_as_nifti(test_output_dirs: OutputFolderForTests, image_type: Any, scale: Any, input_range: Any,
                        output_range: Any) \
        -> None:
    image = np.random.random_sample((dim_z, dim_y, dim_x))
    spacingzyx = (1, 2, 3)
    path_image = test_output_dirs.create_file_or_folder_path(default_image_name)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=spacingzyx)
    io_util.store_as_nifti(image, header, path_image,
                           image_type, scale, input_range, output_range)
    if scale:
        linear_transform = LinearTransform.transform(data=image, input_range=input_range, output_range=output_range)
        image = linear_transform.astype(image_type)  # type: ignore
    assert_nifti_content(test_output_dirs.create_file_or_folder_path(default_image_name),
                         image.shape, header, list(np.unique(image.astype(image_type))), image_type)

    loaded_image = io_util.load_nifti_image(path_image, image_type)
    assert loaded_image.header.spacing == spacingzyx


@pytest.mark.parametrize(["image_type", "scale", "input_range", "output_range"],
                         [(None, None, None, None),
                          (np.ubyte, True, [0, 1], None),
                          (np.short, True, None, [0, 1])])
def test_store_as_nifti_fail(test_output_dirs: OutputFolderForTests, image_type: Any, scale: Any, input_range: Any,
                             output_range: Any) \
        -> None:
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 1, 0, 0, 1, 0, 0), spacing=(1, 2, 4))
    image = np.random.random_sample((dim_z, dim_y, dim_x))
    with pytest.raises(Exception):
        io_util.store_as_nifti(image, header, test_output_dirs.create_file_or_folder_path(default_image_name),
                               image_type, scale, input_range, output_range)


@pytest.mark.parametrize("input_range", [(0, 1), (-1, 1), (0, 255)])
def test_store_as_scaled_ubyte_nifti(test_output_dirs: OutputFolderForTests, input_range: Any) -> None:
    image = np.random.random_sample((dim_z, dim_y, dim_x))
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 2, 4))
    io_util.store_as_scaled_ubyte_nifti(image, header,
                                        test_output_dirs.create_file_or_folder_path(default_image_name),
                                        input_range)
    image = LinearTransform.transform(data=image, input_range=input_range, output_range=(0, 255))
    t = np.unique(image.astype(np.ubyte))
    assert_nifti_content(test_output_dirs.create_file_or_folder_path(default_image_name), image.shape, header, list(t),
                         np.ubyte)


@pytest.mark.parametrize("input_range", [None])
def test_store_as_scaled_ubyte_nifti_fail(test_output_dirs: OutputFolderForTests, input_range: Any) -> None:
    image = np.random.random_sample((dim_z, dim_y, dim_x))
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 2, 4))
    with pytest.raises(Exception):
        io_util.store_as_scaled_ubyte_nifti(image, header,
                                            test_output_dirs.create_file_or_folder_path(default_image_name),
                                            input_range)


def test_store_as_ubyte_nifti(test_output_dirs: OutputFolderForTests) -> None:
    image = np.random.random_sample((dim_z, dim_y, dim_x))
    # get values in [0, 255] range
    image = np.array((image + 1) * 255).astype(int)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 2, 4))
    io_util.store_as_ubyte_nifti(image, header, test_output_dirs.create_file_or_folder_path(default_image_name))
    t = np.unique(image).astype(np.ubyte)
    assert_nifti_content(test_output_dirs.create_file_or_folder_path(default_image_name),
                         image.shape, header, list(t), np.ubyte)


@pytest.mark.parametrize("image",
                         [([[[1]], [[1]], [[1]]]),
                          ([[[0]], [[0]], [[0]]]),
                          ([[[0]], [[1]], [[1]]])])
def test_store_as_binary_nifti(test_output_dirs: OutputFolderForTests, image: Any) -> None:
    image = np.array(image)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 2, 4))
    io_util.store_binary_mask_as_nifti(image, header,
                                       test_output_dirs.create_file_or_folder_path(default_image_name))
    t = np.unique(image)
    assert_nifti_content(test_output_dirs.create_file_or_folder_path(default_image_name), image.shape, header, list(t),
                         np.ubyte)


@pytest.mark.parametrize("image", [([[[0]], [[1]], [[2]]])])
def test_store_as_binary_nifti_fail(test_output_dirs: OutputFolderForTests, image: Any) -> None:
    image = np.array(image)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 1, 0, 0, 1, 0, 0), spacing=(1, 2, 4))
    with pytest.raises(Exception):
        io_util.store_binary_mask_as_nifti(image, header,
                                           test_output_dirs.create_file_or_folder_path(default_image_name))


@pytest.mark.parametrize(["image", "expected"],
                         [([[[1]], [[1]], [[1]]], [255]),
                          ([[[0]], [[0]], [[0]]], [0]),
                          ([[[0.8]], [[0.1]], [[0.4]]], [25, 102, 204])])
def test_store_posteriors_nifti(test_output_dirs: OutputFolderForTests, image: Any, expected: Any) -> None:
    image = np.array(image)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 1, 1))
    io_util.store_posteriors_as_nifti(image, header, test_output_dirs.create_file_or_folder_path(default_image_name))
    assert_nifti_content(test_output_dirs.create_file_or_folder_path(default_image_name),
                         image.shape, header, list(expected), np.ubyte)


@pytest.mark.parametrize("image", [([[[0]], [[1]], [[2]]])])
def test_store_posteriors_nifti_fail(test_output_dirs: OutputFolderForTests, image: Any) -> None:
    image = np.array(image)
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 1, 0, 0, 1, 0, 0), spacing=(1, 1, 1))
    with pytest.raises(Exception):
        io_util.store_posteriors_as_nifti(image, header,
                                          test_output_dirs.create_file_or_folder_path(default_image_name))


def test_store_posteriors_nifti_invalid_entries(test_output_dirs: OutputFolderForTests) -> None:
    image = np.array([0, 1, 2.71, np.nan])
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 1, 0, 0, 1, 0, 0), spacing=(1, 1, 1))
    with pytest.raises(ValueError) as ex:
        io_util.store_posteriors_as_nifti(image, header,
                                          test_output_dirs.create_file_or_folder_path(default_image_name))
    assert "invalid values" in ex.value.args[0]
    assert "2.71" in ex.value.args[0]
    assert "nan" in ex.value.args[0]


@pytest.mark.parametrize(["norm_method", "image_range", "window_level"],
                         [(PhotometricNormalizationMethod.CtWindow, [-100, 100], (40, 50)),
                          (PhotometricNormalizationMethod.CtWindow, [0, 255], (40, 50)),
                          (PhotometricNormalizationMethod.Unchanged, [-1, 1], None),
                          (PhotometricNormalizationMethod.Unchanged, [-40, 40], None)])
def test_store_image_as_short_nifti(test_output_dirs: OutputFolderForTests,
                                    norm_method: PhotometricNormalizationMethod,
                                    image_range: Any,
                                    window_level: Any) -> None:
    window, level = window_level if window_level else (400, 0)

    image = np.random.random_sample((1, 2, 3))
    image_shape = image.shape

    args = SegmentationModelBase(norm_method=norm_method, window=window, level=level, should_validate=False)

    # Get integer values that are in the image range
    image1 = LinearTransform.transform(data=image, input_range=(0, 1), output_range=args.output_range)
    image = image1.astype(np.short)  # type: ignore
    header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 1, 1))
    nifti_name = test_output_dirs.create_file_or_folder_path(default_image_name)
    io_util.store_image_as_short_nifti(image, header, nifti_name, args)

    if norm_method == PhotometricNormalizationMethod.CtWindow:
        output_range = get_range_for_window_level(args.level, args.window)
        image = LinearTransform.transform(data=image, input_range=args.output_range, output_range=output_range)
        image = image.astype(np.short)
    else:
        image = image * 1000

    t = np.unique(image)
    assert_nifti_content(nifti_name, image_shape, header, list(t), np.short)


def test_scale_and_unscale_image(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if an image in the CT value range can be recovered when we save dataset examples
    (undoing the effects of CT Windowing)
    """
    image_size = (5, 5, 5)
    spacing = (1, 2, 3)
    header = ImageHeader(origin=(0, 1, 0), direction=(-1, 0, 0, 0, -1, 0, 0, 0, -1), spacing=spacing)
    np.random.seed(0)
    # Random image values with mean -100, std 100. This will cover a range
    # from -400 to +200 HU
    image = np.random.normal(-100, 100, size=image_size)
    window = 200
    level = -100
    # Lower and upper bounds of the interval of raw CT values that will be retained.
    lower = level - window / 2
    upper = level + window / 2
    # Create a copy of the image with all values outside of the (Window, Level) range set to the boundaries.
    # When saving and loading back in, we will not be able to recover any values that fell outside those boundaries.
    image_restricted = image.copy()
    image_restricted[image < lower] = lower
    image_restricted[image > upper] = upper
    # The image will be saved with voxel type short
    image_restricted = image_restricted.astype(int)
    # Apply window and level, mapping to the usual CNN input value range
    cnn_input_range = (-1, +1)
    image_windowed = LinearTransform.transform(data=image, input_range=(lower, upper), output_range=cnn_input_range)
    args = SegmentationModelBase(norm_method=PhotometricNormalizationMethod.CtWindow, output_range=cnn_input_range,
                                 window=window,
                                 level=level, should_validate=False)

    file_name = test_output_dirs.create_file_or_folder_path("scale_and_unscale_image.nii.gz")
    io_util.store_image_as_short_nifti(image_windowed, header, file_name, args)
    image_from_disk = io_util.load_nifti_image(file_name)
    # noinspection PyTypeChecker
    assert_nifti_content(file_name, image_size, header, np.unique(image_restricted).tolist(), np.short)
    assert np.array_equal(image_from_disk.image, image_restricted)

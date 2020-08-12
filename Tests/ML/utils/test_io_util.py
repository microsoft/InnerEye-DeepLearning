#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Optional, Tuple
from pydicom import Dataset
from pydicom.dataset import FileMetaDataset, FileDataset

import numpy as np
import pytest

from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.dataset.sample import PatientDatasetSource, PatientMetadata
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.dataset_util import DatasetExample, store_and_upload_example
from InnerEye.ML.utils.io_util import ImageHeader, is_nifti_file_path, is_numpy_file_path, \
    load_image_in_known_formats, load_numpy_image, is_dicom_file_path, load_dicom_image
from Tests.ML.util import assert_file_contents
from Tests.fixed_paths_for_tests import full_ml_test_data_path

known_nii_path = full_ml_test_data_path("test_good.nii.gz")
known_array = np.ones((128, 128, 128))
bad_nii_path = full_ml_test_data_path("test_bad.nii.gz")


@pytest.mark.parametrize("path", ["", " ", None, "not_exists", ".", "tests/test_io_util.py"])
def test_bad_path_load_image(path: str) -> None:
    with pytest.raises(ValueError):
        io_util.load_nifti_image(path)


@pytest.mark.parametrize("path", [bad_nii_path])
def test_bad_image_load_image(path: Any) -> None:
    with pytest.raises(ValueError):
        io_util.load_nifti_image(path)


def test_nii_load_image() -> None:
    image_with_header = io_util.load_nifti_image(known_nii_path)
    assert np.array_equal(image_with_header.image, known_array)


@pytest.mark.parametrize("metadata", [None, PatientMetadata(patient_id=0)])
@pytest.mark.parametrize("image_channel", [None, known_nii_path])
@pytest.mark.parametrize("ground_truth_channel", [None, known_nii_path])
@pytest.mark.parametrize("mask_channel", [None, known_nii_path])
def test_load_images_from_dataset_source(
        metadata: Optional[str],
        image_channel: Optional[str],
        ground_truth_channel: Optional[str],
        mask_channel: Optional[str]) -> None:
    """
    Test if images are loaded as expected from channels
    """
    # metadata, image and GT channels must be present. Mask is optional
    if None in [metadata, image_channel, ground_truth_channel]:
        with pytest.raises(Exception):
            _test_load_images_from_channels(metadata, image_channel, ground_truth_channel, mask_channel)
    else:
        _test_load_images_from_channels(metadata, image_channel, ground_truth_channel, mask_channel)


def _test_load_images_from_channels(
        metadata: Any,
        image_channel: Any,
        ground_truth_channel: Any,
        mask_channel: Any) -> None:
    """
    Test if images are loaded as expected from channels
    """
    sample = io_util.load_images_from_dataset_source(
        PatientDatasetSource(
            metadata=metadata,
            image_channels=[image_channel] * 2,
            ground_truth_channels=[ground_truth_channel] * 4,
            mask_channel=mask_channel
        )
    )
    if image_channel:
        image_with_header = io_util.load_nifti_image(image_channel)
        assert list(sample.image.shape) == [2] + list(image_with_header.image.shape)
        assert all([np.array_equal(x, image_with_header.image) for x in sample.image])  # type: ignore
        if mask_channel:
            assert np.array_equal(sample.mask, image_with_header.image)
        if ground_truth_channel:
            assert list(sample.labels.shape) == [5] + list(image_with_header.image.shape)
            assert np.all(sample.labels[0] == 0) and np.all(sample.labels[1:] == 1)


@pytest.mark.parametrize("value, expected",
                         [(["apple"], "apple"),
                          (["apple", "butter"], "apple\nbutter\n")])
def test_save_file(value: Any, expected: Any) -> None:
    file = full_ml_test_data_path("test.txt")
    io_util.save_lines_to_file(Path(file), value)

    assert_file_contents(file, expected)

    os.remove(str(file))


def test_save_dataset_example(test_output_dirs: TestOutputDirectories) -> None:
    """
    Test if the example dataset can be saved as expected.
    """
    image_size = (10, 20, 30)
    label_size = (2,) + image_size
    spacing = (1, 2, 3)
    np.random.seed(0)
    # Image should look similar to what a photonormalized image looks like: Centered around 0
    image = np.random.rand(*image_size) * 2 - 1
    # Labels are expected in one-hot encoding, predictions as class index
    labels = np.zeros(label_size, dtype=int)
    labels[0] = 1
    labels[0, 5:6, 10:11, 15:16] = 0
    labels[1, 5:6, 10:11, 15:16] = 1
    prediction = np.zeros(image_size, dtype=int)
    prediction[4:7, 9:12, 14:17] = 1
    dataset_sample = DatasetExample(epoch=1,
                                    patient_id=2,
                                    header=ImageHeader(origin=(0, 1, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1),
                                                       spacing=spacing),
                                    image=image,
                                    prediction=prediction,
                                    labels=labels)

    images_folder = test_output_dirs.root_dir
    store_and_upload_example(dataset_sample, None, images_folder)
    image_from_disk = io_util.load_nifti_image(os.path.join(images_folder, "p2_e_1_image.nii.gz"))
    labels_from_disk = io_util.load_nifti_image(os.path.join(images_folder, "p2_e_1_label.nii.gz"))
    prediction_from_disk = io_util.load_nifti_image(os.path.join(images_folder, "p2_e_1_prediction.nii.gz"))
    assert image_from_disk.header.spacing == spacing
    # When no photometric normalization is provided when saving, image is multiplied by 1000.
    # It is then rounded to int64, but converted back to float when read back in.
    expected_from_disk = (image * 1000).astype(np.int16).astype(np.float64)
    assert np.array_equal(image_from_disk.image, expected_from_disk)
    assert labels_from_disk.header.spacing == spacing
    assert np.array_equal(labels_from_disk.image, np.argmax(labels, axis=0))
    assert prediction_from_disk.header.spacing == spacing
    assert np.array_equal(prediction_from_disk.image, prediction)


@pytest.mark.parametrize("input", [("foo.txt", False),
                                   ("foo.gz", False),
                                   ("foo.nii.gz", True),
                                   ("foo.nii", True),
                                   ("nii.gz", False),
                                   ])
def test_is_nifti_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_nifti_file_path(file) == expected
    assert is_nifti_file_path(Path(file)) == expected


@pytest.mark.parametrize("input", [("foo.npy", True),
                                   ("foo.mnpy", False),
                                   ("npy", False),
                                   ("foo.txt", False),
                                   ])
def test_is_numpy_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_numpy_file_path(file) == expected
    assert is_numpy_file_path(Path(file)) == expected


def test_load_numpy_image(test_output_dirs: TestOutputDirectories) -> None:
    array_size = (20, 30, 40)
    array = np.ones(array_size)
    assert array.shape == array_size
    npy_file = Path(test_output_dirs.root_dir) / "file.npy"
    assert is_numpy_file_path(npy_file)
    np.save(npy_file, array)
    image = load_numpy_image(npy_file)
    assert image.shape == array_size
    image_and_segmentation = load_image_in_known_formats(npy_file, load_segmentation=False)
    assert image_and_segmentation.images.shape == array_size


@pytest.mark.parametrize("input", [("foo.dcm", True),
                                   ("foo.mdcm", False),
                                   ("dcm", False),
                                   ("foo.txt", False),
                                   ])
def test_is_dicom_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_dicom_file_path(file) == expected
    assert is_dicom_file_path(Path(file)) == expected


def write_test_dicom(array: np.ndarray, path: Path):
    """
    This saves the input array as a Dicom file.
    This function DOES NOT create a usable dicom file and is meant only for testing: tags are set to
    random/default values so that pydicom does not complain when reading the file.
    """
    dicom_dataset = Dataset()
    dicom_metadata = FileMetaDataset()
    dicom_metadata.TransferSyntaxUID = "1.2.840.10008.1.2"
    dicom = FileDataset(path, dicom_dataset, preamble=b"\0"*128, file_meta=dicom_metadata)
    dicom.BitsAllocated = 8
    dicom.Rows = array.shape[0]
    dicom.Columns = array.shape[1]
    dicom.PixelRepresentation = 1
    dicom.SamplesPerPixel = 1
    dicom.PhotometricInterpretation = "MONOCHROME1"
    dicom.PixelData = array
    dicom.save_as(path)


def test_load_dicom_image(test_output_dirs: TestOutputDirectories) -> None:
    array_size = (20, 30)
    array = np.ones(array_size, dtype='uint8')
    array[::2] = 0
    assert array.shape == array_size

    dcm_file = Path(test_output_dirs.root_dir) / "file.dcm"
    assert is_dicom_file_path(dcm_file)
    write_test_dicom(array, dcm_file)

    image = load_dicom_image(dcm_file)
    assert image.ndim == 3 and image.shape[:2] == array_size and image.shape[2] == 1
    assert np.array_equal(image[:, :, 0], array)

    image_and_segmentation = load_image_in_known_formats(dcm_file, load_segmentation=False)
    assert image_and_segmentation.images.ndim == 3 and image_and_segmentation.images.shape[:2] == array_size \
            and image_and_segmentation.images.shape[2] == 1
    assert np.array_equal(image_and_segmentation.images[:, :, 0], array)

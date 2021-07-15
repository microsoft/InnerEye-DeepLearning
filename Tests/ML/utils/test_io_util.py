#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from pathlib import Path
import shutil
from typing import Any, Optional, Tuple
from unittest import mock
import zipfile

import SimpleITK as sitk
import numpy as np
import pydicom
import pytest
import torch
from skimage.transform import resize

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.dataset.sample import PatientDatasetSource, PatientMetadata
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.dataset_util import DatasetExample, store_and_upload_example
from InnerEye.ML.utils.io_util import ImageAndSegmentations, ImageHeader, PhotometricInterpretation, \
    is_dicom_file_path, is_nifti_file_path, is_numpy_file_path, load_dicom_image, load_image_in_known_formats, \
    load_images_and_stack, load_numpy_image, reverse_tuple_float3, load_dicom_series_and_save
from Tests.ML.util import assert_file_contains_string

known_nii_path = full_ml_test_data_path("test_good.nii.gz")
known_array = np.ones((128, 128, 128))
bad_nii_path = full_ml_test_data_path("test_bad.nii.gz")
good_npy_path = full_ml_test_data_path("test_good.npz")
good_h5_path = full_ml_test_data_path("data.h5")
# A sample H&N DICOM series,
dicom_series_folder = full_ml_test_data_path() / "dicom_series_data" / "HN"
# A sample H&N segmentation
HNSEGMENTATION_FILE = full_ml_test_data_path() / "dicom_series_data" / "hnsegmentation.nii.gz"


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


def test_nii_load_zyx(test_output_dirs: OutputFolderForTests) -> None:
    expected_shape = (44, 167, 167)
    file_path = full_ml_test_data_path("patch_sampling/scan_small.nii.gz")
    image: sitk.Image = sitk.ReadImage(str(file_path))
    assert image.GetSize() == reverse_tuple_float3(expected_shape)
    img = sitk.GetArrayFromImage(image)
    assert img.shape == expected_shape
    image_header = io_util.load_nifti_image(file_path)
    assert image_header.image.shape == expected_shape
    assert image_header.header.spacing is not None
    np.testing.assert_allclose(image_header.header.spacing, (3.0, 1.0, 1.0), rtol=0.1)


@pytest.mark.parametrize("metadata", [None, PatientMetadata(patient_id="0")])
@pytest.mark.parametrize("image_channel", [None, known_nii_path, f"{good_h5_path}|volume|0", good_npy_path])
@pytest.mark.parametrize("ground_truth_channel",
                         [None, known_nii_path, f"{good_h5_path}|segmentation|0|1", good_npy_path])
@pytest.mark.parametrize("mask_channel", [None, known_nii_path, good_npy_path])
@pytest.mark.parametrize("check_exclusive", [True, False])
def test_load_images_from_dataset_source(
        metadata: Optional[str],
        image_channel: Optional[str],
        ground_truth_channel: Optional[str],
        mask_channel: Optional[str],
        check_exclusive: bool) -> None:
    """
    Test if images are loaded as expected from channels
    """
    # metadata, image and GT channels must be present. Mask is optional
    if None in [metadata, image_channel, ground_truth_channel]:
        with pytest.raises(Exception):
            _test_load_images_from_channels(metadata, image_channel, ground_truth_channel, mask_channel,
                                            check_exclusive)
    else:
        if check_exclusive:
            with pytest.raises(ValueError) as mutually_exclusive_labels_error:
                _test_load_images_from_channels(metadata, image_channel, ground_truth_channel, mask_channel,
                                                check_exclusive)
            assert 'not mutually exclusive' in str(mutually_exclusive_labels_error.value)
        else:
            _test_load_images_from_channels(metadata, image_channel, ground_truth_channel, mask_channel,
                                            check_exclusive)


def _test_load_images_from_channels(
        metadata: Any,
        image_channel: Any,
        ground_truth_channel: Any,
        mask_channel: Any,
        check_exclusive: bool) -> None:
    """
    Test if images are loaded as expected from channels
    """
    sample = io_util.load_images_from_dataset_source(
        PatientDatasetSource(
            metadata=metadata,
            image_channels=[image_channel] * 2,
            ground_truth_channels=[ground_truth_channel] * 4,
            mask_channel=mask_channel
        ),
        check_exclusive=check_exclusive
    )
    if image_channel:
        image_with_header = io_util.load_image(image_channel)
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
    assert_file_contains_string(file, expected)
    os.remove(str(file))


def test_hdf5_loading() -> None:
    """
    Check that when we access and invalid dataset we get a good exception
    """
    with pytest.raises(ValueError) as valueError:
        io_util.load_image(f"{good_h5_path}|doesnotexist|0|1")
    assert str(good_h5_path) in str(valueError.value)
    assert "doesnotexist" in str(valueError.value)


def test_hdf5_loading_multimap() -> None:
    """
    Check that multimap returns correct image
    """
    image_header = io_util.load_image(f"{good_h5_path}|segmentation|0")
    seg_header = io_util.load_image(f"{good_h5_path}|segmentation|0|1")
    expected = image_header.image == 1
    assert np.array_equal(expected, seg_header.image)


def test_hdf5_loading_multimap_class_do_not_exists() -> None:
    """
    Check that multimap returns correct image if class does not exist
    """
    seg_header = io_util.load_image(f"{good_h5_path}|segmentation|0|555555555555")
    assert np.all(seg_header.image == 0)


def test_save_dataset_example(test_output_dirs: OutputFolderForTests) -> None:
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
    config = SegmentationModelBase(should_validate=False, norm_method=PhotometricNormalizationMethod.Unchanged)
    config.set_output_to(images_folder)
    store_and_upload_example(dataset_sample, config)
    image_from_disk = io_util.load_nifti_image(os.path.join(config.example_images_folder, "p2_e_1_image.nii.gz"))
    labels_from_disk = io_util.load_nifti_image(os.path.join(config.example_images_folder, "p2_e_1_label.nii.gz"))
    prediction_from_disk = io_util.load_nifti_image(os.path.join(config.example_images_folder, "p2_e_1_prediction.nii.gz"))
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


def test_load_numpy_image(test_output_dirs: OutputFolderForTests) -> None:
    array_size = (20, 30, 40)
    array = np.ones(array_size)
    assert array.shape == array_size
    npy_file = test_output_dirs.root_dir / "file.npy"
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


def write_test_dicom(array: np.ndarray, path: Path, is_monochrome2: bool = True,
                     bits_stored: Optional[int] = None) -> None:
    """
    This saves the input array as a Dicom file.
    This function DOES NOT create a usable Dicom file and is meant only for testing: tags are set to
    random/default values so that pydicom does not complain when reading the file.
    """

    # Write a file directly with pydicom is cumbersome (all tags need to be set by hand). Hence using simpleITK to
    # create the file. However SimpleITK does not let you set the tags directly, so using pydicom so set them after.
    image = sitk.GetImageFromArray(array)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path))
    writer.Execute(image)

    ds = pydicom.dcmread(path)
    ds.PhotometricInterpretation = PhotometricInterpretation.MONOCHROME2.value if is_monochrome2 else \
        PhotometricInterpretation.MONOCHROME1.value
    if bits_stored is not None:
        ds.BitsStored = bits_stored
    ds.save_as(path)

@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("is_monochrome2", [True, False])
def test_load_dicom_image_ones(test_output_dirs: OutputFolderForTests,
                               is_signed: bool, is_monochrome2: bool) -> None:
    """
    Test loading of 2D Dicom images filled with binary array of type (uint16) and (int16).
    """
    array_size = (20, 30)
    if not is_signed:
        array = np.ones(array_size, dtype='uint16')
        array[::2] = 0
    else:
        array = -1 * np.ones(array_size, dtype='int16')
        array[::2] = 0

    assert array.shape == array_size

    if is_monochrome2:
        to_write = array
    else:
        if not is_signed:
            to_write = np.zeros(array_size, dtype='uint16')
            to_write[::2] = 1
        else:
            to_write = np.zeros(array_size, dtype='int16')
            to_write[::2] = -1

    dcm_file = test_output_dirs.root_dir / "file.dcm"
    assert is_dicom_file_path(dcm_file)
    write_test_dicom(array=to_write, path=dcm_file, is_monochrome2=is_monochrome2, bits_stored=1)

    image = load_dicom_image(dcm_file)
    assert image.ndim == 2 and image.shape == array_size
    assert np.array_equal(image, array)

    image_and_segmentation = load_image_in_known_formats(dcm_file, load_segmentation=False)
    assert image_and_segmentation.images.ndim == 2 and image_and_segmentation.images.shape == array_size
    assert np.array_equal(image_and_segmentation.images, array)


@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("is_monochrome2", [True, False])
@pytest.mark.parametrize("bits_stored", [14, 16])
def test_load_dicom_image_random(test_output_dirs: OutputFolderForTests,
                                 is_signed: bool, is_monochrome2: bool, bits_stored: int) -> None:
    """
    Test loading of 2D Dicom images of type (uint16) and (int16).
    """
    array_size = (20, 30)
    if not is_signed:
        array = np.random.randint(0, 200, size=array_size, dtype='uint16')
    else:
        array = np.random.randint(-200, 200, size=array_size, dtype='int16')
    assert array.shape == array_size

    if is_monochrome2:
        to_write = array
    else:
        if not is_signed:
            to_write = 2 ** bits_stored - 1 - array
        else:
            to_write = -1 * array - 1

    dcm_file = test_output_dirs.root_dir / "file.dcm"
    assert is_dicom_file_path(dcm_file)
    write_test_dicom(array=to_write, path=dcm_file, is_monochrome2=is_monochrome2, bits_stored=bits_stored)

    image = load_dicom_image(dcm_file)
    assert image.ndim == 2 and image.shape == array_size
    assert np.array_equal(image, array)

    image_and_segmentation = load_image_in_known_formats(dcm_file, load_segmentation=False)
    assert image_and_segmentation.images.ndim == 2 and image_and_segmentation.images.shape == array_size
    assert np.array_equal(image_and_segmentation.images, array)


@pytest.mark.parametrize(["file_path", "expected_shape"],
                         [
                             ("train_and_test_data/id1_mask.nii.gz", (75, 75, 75)),
                             ("hdf5_data/patient_hdf5s/4be9beed-5861-fdd2-72c2-8dd89aadc1ef.h5", (4, 5, 7)),
                         ])
def test_load_image(file_path: str, expected_shape: Tuple) -> None:
    full_file_path = full_ml_test_data_path() / file_path
    image_and_segmentation = load_image_in_known_formats(full_file_path, load_segmentation=False)
    assert image_and_segmentation.images.shape == expected_shape


@pytest.mark.parametrize(["file_path_str", "expected_shape"],
                         [
                             ("train_and_test_data/id1_mask.nii.gz", (75, 75, 75)),
                             ("hdf5_data/patient_hdf5s/4be9beed-5861-fdd2-72c2-8dd89aadc1ef.h5", (4, 5, 7)),
                         ])
def test_load_and_stack(file_path_str: str, expected_shape: Tuple) -> None:
    file_path = Path(file_path_str)
    files = [full_ml_test_data_path() / f for f in [file_path, file_path]]
    stacked = load_images_and_stack(files, load_segmentation=False)
    assert torch.is_tensor(stacked.segmentations)
    assert stacked.segmentations is not None
    assert stacked.segmentations.shape == (0,)
    assert torch.is_tensor(stacked.images)
    assert stacked.images.shape == (2,) + expected_shape


def test_load_and_stack_with_segmentation() -> None:
    expected_shape = (4, 5, 7)
    file_path = full_ml_test_data_path() / "hdf5_data/patient_hdf5s/4be9beed-5861-fdd2-72c2-8dd89aadc1ef.h5"
    files = [file_path, file_path]
    stacked = load_images_and_stack(files, load_segmentation=True)
    assert stacked.segmentations is not None
    assert torch.is_tensor(stacked.segmentations)
    assert stacked.segmentations.dtype == torch.uint8
    assert stacked.segmentations.shape == (2,) + expected_shape
    assert torch.is_tensor(stacked.images)
    assert stacked.images.dtype == torch.float16
    assert stacked.images.shape == (2,) + expected_shape


def test_load_and_stack_with_crop() -> None:
    image_size = (10, 12, 14)
    crop_shape = (4, 6, 8)
    center_start = (3, 3, 3)
    assert np.allclose(np.array(center_start) * 2 + np.array(crop_shape), np.array(image_size))
    # Create a fake image that is all zeros apart from ones in the center, right where we expect the
    # center crop to be taken from. We can later assert that the right crop was taken by checking that only
    # values of 1.0 are in the image
    image = np.zeros(image_size)
    image[center_start[0]:center_start[0] + crop_shape[0],
    center_start[1]:center_start[1] + crop_shape[1],
    center_start[2]:center_start[2] + crop_shape[2]] = 1
    segmentation = image * 2
    mock_return = ImageAndSegmentations(image, segmentation)
    with mock.patch("InnerEye.ML.utils.io_util.load_image_in_known_formats", return_value=mock_return):
        stacked = load_images_and_stack([Path("doesnotmatter")], load_segmentation=True,
                                        center_crop_size=crop_shape)
        assert torch.is_tensor(stacked.images)
        assert stacked.images.shape == (1,) + crop_shape
        assert torch.all(stacked.images == 1.0)
        assert torch.is_tensor(stacked.segmentations)
        assert stacked.segmentations is not None
        assert stacked.segmentations.shape == (1,) + crop_shape
        assert torch.all(stacked.segmentations == 2.0)


def test_load_images_when_empty() -> None:
    stacked = load_images_and_stack([], load_segmentation=False)
    assert stacked.images.shape == (0,)
    assert stacked.segmentations is not None
    assert stacked.segmentations.shape == (0,)


def test_load_images_and_stack_2d_ones(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load of 2D images filled with (int) ones.
    """
    image_size = (20, 30)

    array = np.ones(image_size, dtype='uint16')
    write_test_dicom(array, test_output_dirs.root_dir / "file1.dcm")
    write_test_dicom(array, test_output_dirs.root_dir / "file2.dcm")
    write_test_dicom(array, test_output_dirs.root_dir / "file3.dcm")

    expected_tensor = torch.from_numpy(np.ones((3, 1) + image_size))

    file_list = [test_output_dirs.root_dir / f"file{i}.dcm" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=(1,) + image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1] == 1
    assert imaging_data.images.shape[2:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_2d_random(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load of 2D images
    """
    image_size = (20, 30)
    low = 0
    high = 200

    array1 = np.random.randint(low=low, high=high, size=image_size, dtype='uint16')
    write_test_dicom(array1, test_output_dirs.root_dir / "file1.dcm")
    array2 = np.random.randint(low=low, high=high, size=image_size, dtype='uint16')
    write_test_dicom(array2, test_output_dirs.root_dir / "file2.dcm")
    array3 = np.random.randint(low=low, high=high, size=image_size, dtype='uint16')
    write_test_dicom(array3, test_output_dirs.root_dir / "file3.dcm")

    expected_tensor = torch.from_numpy(np.expand_dims(np.stack([array1, array2, array3]).astype(float), axis=1))

    file_list = [test_output_dirs.root_dir / f"file{i}.dcm" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=(1,) + image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1] == 1
    assert imaging_data.images.shape[2:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_2d_with_resize_ones(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load and resize of 2D images filled with (int) ones.
    """
    image_size = (20, 30)

    array = np.ones((10, 20), dtype='uint16')
    write_test_dicom(array, test_output_dirs.root_dir / "file1.dcm")
    array = np.ones((20, 30), dtype='uint16')
    write_test_dicom(array, test_output_dirs.root_dir / "file2.dcm")
    array = np.ones((30, 10), dtype='uint16')
    write_test_dicom(array, test_output_dirs.root_dir / "file3.dcm")

    expected_tensor = torch.from_numpy(np.ones((3, 1) + image_size))

    file_list = [test_output_dirs.root_dir / f"file{i}.dcm" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=(1,) + image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1] == 1
    assert imaging_data.images.shape[2:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_2d_with_resize_random(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load and resize of 2D images
    """
    image_size = (20, 30)
    low = 0
    high = 200

    array1 = np.random.randint(low=low, high=high, size=(10, 20), dtype='uint16')
    write_test_dicom(array1, test_output_dirs.root_dir / "file1.dcm")
    array2 = np.random.randint(low=low, high=high, size=(20, 30), dtype='uint16')
    write_test_dicom(array2, test_output_dirs.root_dir / "file2.dcm")
    array3 = np.random.randint(low=low, high=high, size=(30, 20), dtype='uint16')
    write_test_dicom(array3, test_output_dirs.root_dir / "file3.dcm")

    array1 = resize(array1.astype(np.float), image_size, anti_aliasing=True)
    array3 = resize(array3.astype(np.float), image_size, anti_aliasing=True)

    expected_tensor = torch.from_numpy(np.expand_dims(np.stack([array1, array2, array3]).astype(float), axis=1))

    file_list = [test_output_dirs.root_dir / f"file{i}.dcm" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=(1,) + image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1] == 1
    assert imaging_data.images.shape[2:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_3d_with_resize_ones(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load and resize of 3D images filled with (float) ones.
    """
    image_size = (20, 30, 20)

    array = np.ones((10, 20, 10))
    np.save(test_output_dirs.root_dir / "file1.npy", array)
    array = np.ones((20, 30, 20))
    np.save(test_output_dirs.root_dir / "file2.npy", array)
    array = np.ones((30, 10, 30))
    np.save(test_output_dirs.root_dir / "file3.npy", array)

    expected_tensor = torch.from_numpy(np.ones((3,) + image_size))

    file_list = [test_output_dirs.root_dir / f"file{i}.npy" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_3d_with_resize_random(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test load and resize of 3D images
    """
    image_size = (20, 30, 20)
    low = -200
    high = 200

    array1 = np.random.randint(low=low, high=high, size=(10, 20, 10)).astype(np.float)
    np.save(test_output_dirs.root_dir / "file1.npy", array1)
    array2 = np.random.randint(low=low, high=high, size=(20, 30, 20)).astype(np.float)
    np.save(test_output_dirs.root_dir / "file2.npy", array2)
    array3 = np.random.randint(low=low, high=high, size=(30, 10, 30)).astype(np.float)
    np.save(test_output_dirs.root_dir / "file3.npy", array3)

    array1 = resize(array1.astype(np.float), image_size, anti_aliasing=True)
    array3 = resize(array3.astype(np.float), image_size, anti_aliasing=True)

    expected_tensor = torch.from_numpy(np.stack([array1, array2, array3]).astype(float))

    file_list = [test_output_dirs.root_dir / f"file{i}.npy" for i in range(1, 4)]
    imaging_data = load_images_and_stack(file_list,
                                         load_segmentation=False,
                                         image_size=image_size)

    assert len(imaging_data.images.shape) == 4
    assert imaging_data.images.shape[0] == 3
    assert imaging_data.images.shape[1:] == image_size
    assert torch.allclose(imaging_data.images, expected_tensor)


def test_load_images_and_stack_with_resize_only_float(test_output_dirs: OutputFolderForTests) -> None:
    """
    Don't allow int type images to be loaded if image_size is set:
    skimage.transform.resize will not resize these correctly
    """
    image_size = (20, 30, 20)

    array = np.ones((10, 20, 20), dtype='uint16')
    np.save(test_output_dirs.root_dir / "file.npy", array)
    file_list = [test_output_dirs.root_dir / "file.npy"]

    with pytest.raises(ValueError):
        load_images_and_stack(file_list,
                              load_segmentation=False,
                              image_size=image_size)


def test_load_dicom_series(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a DICOM series can be loaded.

    :param test_output_dirs: Test output directories.
    :return: None.
    """
    nifti_file = test_output_dirs.root_dir / "test_dicom_series.nii.gz"
    load_dicom_series_and_save(dicom_series_folder, nifti_file)
    expected_shape = (3, 512, 512)
    image_header = io_util.load_nifti_image(nifti_file)
    assert image_header.image.shape == expected_shape
    assert image_header.header.spacing is not None
    np.testing.assert_allclose(image_header.header.spacing, (2.5, 1.269531, 1.269531), rtol=0.1)


def zip_known_dicom_series(zip_file_path: Path) -> None:
    """
    Create a zipped reference DICOM series.

    Zip the reference DICOM series from test_data into zip_file_path.

    :param zip_file_path: Target zip file.
    """
    zip_file_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_file_path.with_suffix('')), 'zip', str(dicom_series_folder))


def test_create_dicom_series(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a DICOM series can be created.

    :param test_output_dirs: Test output directories.
    :return: None.
    """
    test_shape = (24, 36, 48)  # (#slices, #rows, #columns)
    test_spacing = (1.0, 1.0, 2.5)  # (column spacing, row spacing, slice spacing)
    series_folder = test_output_dirs.root_dir / "series"
    series_folder.mkdir()
    # Create the series
    image_data = io_util.create_dicom_series(series_folder, test_shape, test_spacing)
    # Load it in
    loaded_image = io_util.load_dicom_series(series_folder)
    # GetSize returns (width, height, depth)
    assert loaded_image.GetSize() == reverse_tuple_float3(test_shape)
    assert loaded_image.GetSpacing() == test_spacing
    # Get the data from the loaded series and compare it.
    loaded_image_data = sitk.GetArrayFromImage(loaded_image).astype(np.float)
    assert loaded_image_data.shape == test_shape
    # Data is saved 16 bit, so need a generous tolerance.
    assert np.allclose(loaded_image_data, image_data, atol=1e-1)


def test_zip_random_dicom_series(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a DICOM series can be created.
        :param test_output_dirs: Test output directories.
    :return: None.
    """
    test_shape = (24, 36, 48)  # (#slices, #rows, #columns)
    test_spacing = (1.0, 1.0, 2.5)  # (column spacing, row spacing, slice spacing)
    zip_file_path = test_output_dirs.root_dir / "pack" / "random.zip"
    scratch_folder = test_output_dirs.root_dir / "scratch"
    io_util.zip_random_dicom_series(test_shape, test_spacing, zip_file_path, scratch_folder)

    assert zip_file_path.is_file()
    series_folder = test_output_dirs.root_dir / "unpack"
    with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
        zip_file.extractall(series_folder)
    # Load it in
    loaded_image = io_util.load_dicom_series(series_folder)
    # GetSize returns (width, height, depth)
    assert loaded_image.GetSize() == reverse_tuple_float3(test_shape)
    assert loaded_image.GetSpacing() == test_spacing

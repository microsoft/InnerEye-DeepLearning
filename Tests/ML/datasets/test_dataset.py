#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd
import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.ML.config import PaddingMode, SegmentationModelBase
from InnerEye.ML.dataset.cropping_dataset import CroppingDataset
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset, collate_with_metadata
from InnerEye.ML.dataset.sample import CroppedSample, PatientMetadata, SAMPLE_METADATA_FIELD, \
    Sample, SegmentationSampleBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.utils import image_util, ml_util
from InnerEye.ML.utils.io_util import ImageDataType
from InnerEye.ML.utils.transforms import Compose3D
from Tests.Common.test_util import full_ml_test_data_path
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import DummyPatientMetadata, load_train_and_test_data_channels

crop_size = [55, 55, 55]

@pytest.fixture
def num_dataload_workers() -> int:
    """PyTorch support for multiple dataloader workers is flaky on Windows (so return 0)"""
    return 4 if common_util.is_linux() else 0


@pytest.fixture
def default_config() -> ModelConfigBase:
    config = DummyModel()
    config.set_output_to(str(full_ml_test_data_path("outputs")))
    return config


# create random test gt image image to sample from
@pytest.fixture
def gt_image(default_config: SegmentationModelBase) -> Union[int, np.ndarray]:
    np.random.seed(1)
    gt_image = np.random.randint(0, default_config.number_of_classes, (75, 75, 75))
    return gt_image


@pytest.fixture
def large_random_gt_image(default_config: SegmentationModelBase) -> Union[int, np.ndarray]:
    np.random.seed(2)
    gt_image = np.random.randint(0, default_config.number_of_classes, (200, 200, 200))
    return gt_image


@pytest.fixture
def large_random_input_image() -> Any:
    np.random.seed(3)
    gt_image = np.random.uniform(-1, 1, (200, 200, 200))
    return gt_image


@pytest.fixture
def random_crop_centre_for_large_image() -> Any:
    np.random.seed(4)
    random_centre = np.random.randint(crop_size[0], (200 - crop_size[0]))
    return random_centre, random_centre, random_centre


@pytest.fixture
def input_image() -> Any:
    np.random.seed(5)
    input_image = np.random.uniform(-1, 1, (crop_size[0], crop_size[1], crop_size[2], 1))
    return input_image


@pytest.fixture
def large_random_mask() -> Union[int, np.ndarray]:
    np.random.seed(6)
    # creates 0 and 1, 2 is excluded!
    mask = np.random.randint(0, 2, (200, 200, 200))
    return mask


@pytest.fixture
def random_image_crop() -> Any:
    np.random.seed(7)
    return np.random.uniform(-1, 1, crop_size)


@pytest.fixture
def random_label_crop(default_config: SegmentationModelBase) -> Any:
    np.random.seed(8)
    return np.random.uniform(0, default_config.number_of_classes, crop_size)


@pytest.fixture
def random_mask_crop() -> Union[int, np.ndarray]:
    np.random.seed(9)
    return np.random.randint(0, 2, crop_size)


@pytest.fixture
def random_patient_id() -> Union[int, np.ndarray]:
    np.random.seed(10)
    return np.random.randint(0, 10000)


@pytest.fixture
def cropping_dataset(default_config: SegmentationModelBase,
                     normalize_fn: Callable) -> CroppingDataset:
    df = default_config.get_dataset_splits()
    return CroppingDataset(args=default_config, data_frame=df.train)


@pytest.fixture
def full_image_dataset(default_config: SegmentationModelBase,
                       normalize_fn: Callable) -> FullImageDataset:
    df = default_config.get_dataset_splits()
    return FullImageDataset(args=default_config,
                            full_image_sample_transforms=Compose3D([normalize_fn]),  # type: ignore
                            data_frame=df.train)


@pytest.fixture
def full_image_dataset_no_mask(default_config: SegmentationModelBase,
                               normalize_fn: Callable) -> FullImageDataset:
    default_config.mask_id = None
    df = default_config.get_dataset_splits()
    return FullImageDataset(args=default_config,
                            full_image_sample_transforms=Compose3D([normalize_fn]),  # type: ignore
                            data_frame=df.train)


@pytest.fixture
def normalize_fn(default_config: SegmentationModelBase) -> PhotometricNormalization:
    return PhotometricNormalization(default_config)


def test_dataset_content(default_config: ModelConfigBase, gt_image: np.ndarray,
                         cropping_dataset: CroppingDataset, full_image_dataset: FullImageDataset) -> None:
    # check number of patients
    assert len(full_image_dataset) == len(cropping_dataset) == 2
    assert len(np.unique(gt_image)) == default_config.number_of_classes


def test_sample(random_image_crop: Any, random_mask_crop: Any, random_label_crop: Any, random_patient_id: Any) -> None:
    """
    Tests that after creating and extracting a sample we obtain the same result
    :return:
    """
    metadata = PatientMetadata(patient_id='42', institution="foo")
    sample = Sample(image=random_image_crop,
                    mask=random_mask_crop,
                    labels=random_label_crop,
                    metadata=metadata)

    patched_sample = CroppedSample(image=random_image_crop,
                                   mask=random_mask_crop,
                                   labels=random_label_crop,
                                   mask_center_crop=random_mask_crop,
                                   labels_center_crop=random_label_crop,
                                   metadata=metadata,
                                   center_indices=np.zeros((1, 3)))

    extracted_sample = sample.get_dict()
    extracted_patched_sample = patched_sample.get_dict()

    sample_and_patched_sample_equal: Callable[[str, Any], bool] \
        = lambda k, x: bool(
        np.array_equal(extracted_sample[k], extracted_patched_sample[k]) and np.array_equal(extracted_patched_sample[k],
                                                                                            x))

    assert sample_and_patched_sample_equal("image", random_image_crop)
    assert sample_and_patched_sample_equal("mask", random_mask_crop)
    assert sample_and_patched_sample_equal("labels", random_label_crop)

    assert np.array_equal(extracted_patched_sample["mask_center_crop"], random_mask_crop)
    assert np.array_equal(extracted_patched_sample["labels_center_crop"], random_label_crop)
    assert extracted_sample["metadata"] == extracted_patched_sample["metadata"] == metadata


def test_cropping_dataset_as_data_loader(cropping_dataset: CroppingDataset, num_dataload_workers: int) -> None:
    batch_size = 2
    loader = cropping_dataset.as_data_loader(shuffle=True, batch_size=batch_size,
                                             num_dataload_workers=num_dataload_workers)
    for i, item in enumerate(loader):
        item = CroppedSample.from_dict(sample=item)
        assert item is not None
        assert item.image.shape == \
               (batch_size,
                cropping_dataset.args.number_of_image_channels) + cropping_dataset.args.crop_size  # type: ignore
        assert item.mask.shape == (batch_size,) + cropping_dataset.args.crop_size  # type: ignore
        assert item.labels.shape == \
               (batch_size, cropping_dataset.args.number_of_classes) + cropping_dataset.args.crop_size  # type: ignore
        # check the mask center crops are as expected
        assert item.mask_center_crop.shape == (batch_size,) + cropping_dataset.args.center_size  # type: ignore
        assert item.labels_center_crop.shape == \
               (batch_size, cropping_dataset.args.number_of_classes) + cropping_dataset.args.center_size  # type: ignore

        # check the contents of the center crops
        for b in range(batch_size):
            expected = image_util.get_center_crop(image=item.mask[b], crop_shape=cropping_dataset.args.center_size)
            assert np.array_equal(item.mask_center_crop[b], expected)

            for c in range(len(item.labels_center_crop[b])):
                expected = image_util.get_center_crop(image=item.labels[b][c],
                                                      crop_shape=cropping_dataset.args.center_size)
                assert np.array_equal(item.labels_center_crop[b][c], expected)


def test_cropping_dataset_sample_dtype(cropping_dataset: CroppingDataset, num_dataload_workers: int) -> None:
    """
    Tests the data type of torch tensors (e.g. image, labels, and mask) created by the dataset generator,
    which are provided as input into the computational graph
    :return:
    """
    loader = cropping_dataset.as_data_loader(shuffle=True, batch_size=2,
                                             num_dataload_workers=num_dataload_workers)
    for i, item in enumerate(loader):
        item = CroppedSample.from_dict(item)
        assert item.image.numpy().dtype == ImageDataType.IMAGE.value
        assert item.labels.numpy().dtype == ImageDataType.SEGMENTATION.value
        assert item.mask.numpy().dtype == ImageDataType.MASK.value
        assert item.mask_center_crop.numpy().dtype == ImageDataType.MASK.value
        assert item.labels_center_crop.numpy().dtype == ImageDataType.SEGMENTATION.value


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
def test_cropping_dataset_padding(cropping_dataset: CroppingDataset, num_dataload_workers: int) -> None:
    """
    Tests the data type of torch tensors (e.g. image, labels, and mask) created by the dataset generator,
    which are provided as input into the computational graph
    :return:
    """
    cropping_dataset.args.crop_size = (300, 300, 300)
    cropping_dataset.args.padding_mode = PaddingMode.Zero
    loader = cropping_dataset.as_data_loader(shuffle=True, batch_size=2, num_dataload_workers=1)

    for i, item in enumerate(loader):
        sample = CroppedSample.from_dict(item)
        assert sample.image.shape[-3:] == cropping_dataset.args.crop_size


def test_cropping_dataset_has_reproducible_randomness(cropping_dataset: CroppingDataset,
                                                      num_dataload_workers: int) -> None:
    cropping_dataset.dataset_indices = ['1', '2'] * 2
    expected_center_indices = None
    for k in range(3):
        ml_util.set_random_seed(1)
        loader = cropping_dataset.as_data_loader(shuffle=True, batch_size=4,
                                                 num_dataload_workers=num_dataload_workers)
        for i, item in enumerate(loader):
            item = CroppedSample.from_dict(sample=item)
            if expected_center_indices is None:
                expected_center_indices = item.center_indices
            else:
                assert np.array_equal(expected_center_indices, item.center_indices)


def test_csv_dataset_as_data_loader(normalize_fn: Any,
                                    full_image_dataset: FullImageDataset, num_dataload_workers: int) -> None:
    batch_size = 2
    # load the original images separately for comparison
    expected_samples = load_train_and_test_data_channels(patient_ids=list(range(1, batch_size + 1)),
                                                         normalization_fn=normalize_fn)
    csv_dataset_loader = full_image_dataset.as_data_loader(batch_size=batch_size, shuffle=True,
                                                           num_dataload_workers=num_dataload_workers)
    for i, batch in enumerate(csv_dataset_loader):
        for x in range(batch_size):
            actual_sample = {}
            for k, v in batch.items():
                actual_sample[k] = v[x]
            sample = Sample.from_dict(sample=actual_sample)
            # have to do this as the ordering in which the dataloader gives samples is non-deterministic
            expected_sample = expected_samples[sample.patient_id - 1]  # type: ignore

            assert sample.patient_id == expected_sample.patient_id
            assert np.array_equal(sample.image, expected_sample.image)
            assert np.array_equal(sample.labels, expected_sample.labels)
            assert np.array_equal(sample.mask, expected_sample.mask)


def test_full_image_dataset_no_mask(full_image_dataset_no_mask: FullImageDataset) -> None:
    assert np.all(np.array([Sample.from_dict(sample=x).mask for x in full_image_dataset_no_mask]) == 1)  # type: ignore


@pytest.mark.parametrize("crop_size", [(4, 4, 4), (8, 6, 4)])
def test_create_possibly_padded_sample_for_cropping(crop_size: Any) -> None:
    image_size = [4] * 3
    image = np.random.uniform(size=[1] + image_size)
    labels = np.zeros(shape=[2] + image_size)
    mask = np.zeros(shape=image_size, dtype=ImageDataType.MASK.value)

    cropped_sample = CroppingDataset.create_possibly_padded_sample_for_cropping(
        sample=Sample(image=image, labels=labels, mask=mask, metadata=DummyPatientMetadata),
        crop_size=crop_size,
        padding_mode=PaddingMode.Zero
    )

    assert cropped_sample.image.shape[-3:] == crop_size
    assert cropped_sample.labels.shape[-3:] == crop_size
    assert cropped_sample.mask.shape[-3:] == crop_size


@pytest.mark.parametrize("use_mask", [False, True])
def test_cropped_sample(use_mask: bool) -> None:
    ml_util.set_random_seed(1)
    image_size = [4] * 3
    crop_size = (2, 2, 2)
    center_size = (1, 1, 1)

    # create small image sample for random cropping
    image = np.random.uniform(size=[1] + image_size)
    labels = np.zeros(shape=[2] + image_size)
    # Two foreground points in the corners at (0, 0, 0) and (3, 3, 3)
    labels[0] = 1
    labels[0, 0, 0, 0] = 0
    labels[0, 3, 3, 3] = 0
    labels[1, 0, 0, 0] = 1
    labels[1, 3, 3, 3] = 1
    crop_slicer: Optional[slice]
    if use_mask:
        # If mask is used, the cropping center point should be inside the mask.
        # Create a mask that has exactly 1 point of overlap with the labels,
        # that point must then be the center
        mask = np.zeros(shape=image_size, dtype=ImageDataType.MASK.value)
        mask[3, 3, 3] = 1
        expected_center: Optional[List[int]] = [3, 3, 3]
        crop_slicer = slice(2, 4)
    else:
        mask = np.ones(shape=image_size, dtype=ImageDataType.MASK.value)
        expected_center = None
        crop_slicer = None

    sample = Sample(
        image=image,
        labels=labels,
        mask=mask,
        metadata=DummyPatientMetadata
    )

    for _ in range(0, 100):
        cropped_sample = CroppingDataset.create_random_cropped_sample(
            sample=sample,
            crop_size=crop_size,
            center_size=center_size,
            class_weights=[0, 1]
        )

        if expected_center is not None:
            assert list(cropped_sample.center_indices) == expected_center  # type: ignore
            assert np.array_equal(cropped_sample.image, sample.image[:, crop_slicer, crop_slicer, crop_slicer])
            assert np.array_equal(cropped_sample.labels, sample.labels[:, crop_slicer, crop_slicer, crop_slicer])
            assert np.array_equal(cropped_sample.mask, sample.mask[crop_slicer, crop_slicer, crop_slicer])
        else:
            # The crop center point must be any point that has a positive foreground label
            center = cropped_sample.center_indices
            print("Center point chosen: {}".format(center))
            assert labels[1, center[0], center[1], center[2]] != 0


def test_restricted_dataset(default_config: ModelConfigBase, ) -> None:
    default_config.restrict_subjects = None
    splits = default_config.get_dataset_splits()
    assert all([len(x) > 1 for x in
                [splits.train.subject.unique(), splits.test.subject.unique(), splits.val.subject.unique()]])
    default_config.restrict_subjects = "1"
    splits = default_config.get_dataset_splits()
    assert all([len(x) == 1 for x in
                [splits.train.subject.unique(), splits.test.subject.unique(), splits.val.subject.unique()]])


def test_patient_metadata() -> None:
    """
    Loading a dataset where all patient metadata columns are present
    :return:
    """
    file = full_ml_test_data_path("dataset_with_full_header.csv")
    df = pd.read_csv(file, dtype=str)
    subject = "511"
    expected_institution = "85aaee5f-f5f3-4eae-b6cd-26b0070156d8"
    expected_series = "22ef9c5e149650f9cb241d1aa622ad1731b91d1a1df770c05541228b47845ae4"
    expected_tags = "FOO;BAR"
    metadata = PatientMetadata.from_dataframe(df, subject)
    assert metadata is not None
    assert metadata.patient_id == subject
    assert metadata.institution == expected_institution
    assert metadata.series == expected_series
    assert metadata.tags_str == expected_tags

    # Now modify the dataset such that there is no single value for tags. Tags should no longer be
    # populated, but the other fields should be.
    df['tags'] = ["something", ""]
    metadata = PatientMetadata.from_dataframe(df, subject)
    assert metadata.series == expected_series
    assert metadata.institution == expected_institution
    assert metadata.tags_str is None


def test_min_patient_metadata() -> None:
    """
    Loading a dataset where only required columns are present
    """
    df = pd.read_csv(full_ml_test_data_path("dataset.csv"), dtype=str)
    df = df.drop(columns="institutionId")
    patient_id = "1"
    metadata = PatientMetadata.from_dataframe(df, patient_id)
    assert metadata.patient_id == patient_id
    assert metadata.series is None
    assert metadata.institution is None
    assert metadata.tags_str is None


def test_get_all_metadata(default_config: ModelConfigBase) -> None:
    df = default_config.get_dataset_splits().train
    assert PatientMetadata.from_dataframe(df, '1') == PatientMetadata(patient_id='1', institution="1")
    assert PatientMetadata.from_dataframe(df, '2') == PatientMetadata(patient_id='2', institution="2")


def test_sample_metadata_field() -> None:
    """
    Test that the string constant we use to identify the metadata field is really matching the
    field name in SampleWithMetadata
    """
    s = SegmentationSampleBase(metadata=DummyPatientMetadata)
    fields = vars(s)
    assert len(fields) == 1
    assert SAMPLE_METADATA_FIELD in fields


def test_custom_collate() -> None:
    """
    Tests the custom collate function that collates metadata into lists.
    """
    metadata = PatientMetadata(patient_id='42')
    foo = "foo"
    d1 = {foo: 1, SAMPLE_METADATA_FIELD: "something"}
    d2 = {foo: 2, SAMPLE_METADATA_FIELD: metadata}
    result = collate_with_metadata([d1, d2])
    assert foo in result
    assert SAMPLE_METADATA_FIELD in result
    assert isinstance(result[SAMPLE_METADATA_FIELD], list)
    assert result[SAMPLE_METADATA_FIELD] == ["something", metadata]
    assert isinstance(result[foo], torch.Tensor)
    assert result[foo].tolist() == [1, 2]


def test_sample_construct_copy(random_image_crop: Any, random_mask_crop: Any, random_label_crop: Any) -> None:
    sample = Sample(
        image=random_image_crop,
        mask=random_mask_crop,
        labels=random_label_crop,
        metadata=PatientMetadata(patient_id='1')
    )

    sample_clone = sample.clone_with_overrides()
    assert sample.get_dict() == sample_clone.get_dict()
    assert type(sample) == type(sample_clone)

    sample_clone = sample.clone_with_overrides(metadata=PatientMetadata(patient_id='2'))
    assert sample_clone.patient_id == 2

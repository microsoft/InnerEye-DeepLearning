#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Optional

import pytest

from InnerEye.Common.type_annotations import IntOrTuple3, TupleInt3
from InnerEye.ML.config import ModelArchitectureConfig, SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel, CropSizeConstraints
from InnerEye.ML.models.architectures.unet_3d import UNet3D
from InnerEye.ML.utils.model_util import build_net
from Tests.ML.pipelines.test_forward_pass import SimpleModel


def test_crop_size_constraints() -> None:
    """
    Test the basic logic to validate a crop size inside of a CropSizeConstraints instance.
    """

    def check(crop_size: TupleInt3,
              multiple_of: Optional[IntOrTuple3] = None,
              minimum: Optional[IntOrTuple3] = None) -> CropSizeConstraints:
        constraints = CropSizeConstraints(minimum_size=minimum, multiple_of=multiple_of)
        constraints.validate(crop_size)
        return constraints

    # crop_size_multiple == 1: Any crop size is allowed.
    c = check((9, 10, 11), multiple_of=1)
    # If a scalar multiple_of is used, it should be stored expanded along dimensions
    assert c.multiple_of == (1, 1, 1)
    # Using a tuple multiple_of: Crop is twice as large as multiple_of in each dimension, this is hence valid.
    c = check((10, 12, 14), multiple_of=(5, 6, 7))
    assert c.multiple_of == (5, 6, 7)
    # Minimum size has not been provided, should default to multiple_of.
    assert c.minimum_size == (5, 6, 7)
    # Try with a couple more common crop sizes
    check((32, 64, 64), multiple_of=16)
    check((1, 64, 64), multiple_of=(1, 16, 16))
    # Crop size is at the allowed minimum: This is valid
    check((3, 4, 5), multiple_of=(3, 4, 5))
    # Provide a scalar minimum: Should be stored expanded into 3 dimensions
    c = check((9, 10, 11), minimum=2)
    assert c.minimum_size == (2, 2, 2)
    assert c.multiple_of is None
    # Provide a tuple minimum
    c = check((9, 10, 11), minimum=(5, 6, 7))
    assert c.minimum_size == (5, 6, 7)
    # A crop size at exactly the minimum is valid
    check((5, 6, 7), minimum=(5, 6, 7))
    # Checking for minimum and multiple at the same time
    check((9, 10, 11), minimum=1, multiple_of=1)
    check((10, 12, 14), minimum=(5, 6, 7), multiple_of=2)

    def assert_fails(crop_size: TupleInt3,
                     multiple_of: Optional[IntOrTuple3] = None,
                     minimum: Optional[IntOrTuple3] = None) -> None:
        with pytest.raises(ValueError) as err:
            check(crop_size, multiple_of, minimum)
        assert str(crop_size) in str(err)
        assert "Crop size is not valid" in str(err)

    # Crop size is not a multiple of 2 in dimensions 0 and 1
    assert_fails((3, 4, 5), 2)
    # Crop size is not a multiple of 6 in dimension 2
    assert_fails((3, 4, 5), (3, 4, 6))
    assert_fails((16, 16, 200), 16)
    # Crop size is too small
    assert_fails((1, 2, 3), (10, 10, 10))
    assert_fails((10, 20, 30), multiple_of=10, minimum=20)

    # Minimum size must be at least as large as multiple_of:
    with pytest.raises(ValueError) as err:
        CropSizeConstraints(minimum_size=10, multiple_of=20, num_dimensions=2)
    assert "minimum size must be at least as large as the multiple_of" in str(err)
    assert "(10, 10)" in str(err)
    assert "(20, 20)" in str(err)

    # Check that num_dimensions is working as expected (though not presently used in the codebase)
    c = CropSizeConstraints(minimum_size=30, multiple_of=10, num_dimensions=2)
    assert c.multiple_of == (10, 10)
    assert c.minimum_size == (30, 30)


def test_crop_size_constructor() -> None:
    """
    Test error handling in the constructor of CropSizeConstraints
    """
    # Minimum size is given as 3-tuple, but working with 2 dimensions:
    with pytest.raises(ValueError) as err:
        CropSizeConstraints(minimum_size=(1, 2, 3), num_dimensions=2)
    assert "must have length 2" in str(err)
    # Minimum size and multiple_of are not compatible:
    with pytest.raises(ValueError) as err:
        CropSizeConstraints(minimum_size=(1, 2, 3), multiple_of=16)
    assert "The minimum size must be at least as large" in str(err)


def test_crop_size_check_in_model() -> None:
    """
    Test that crop size checks are correctly integrated and called from within BaseModel
    """

    def create_model(crop_size: TupleInt3, multiple_of: IntOrTuple3) -> BaseSegmentationModel:
        model = SimpleModel(1, [1], 2, 2, crop_size_constraints=CropSizeConstraints(multiple_of=multiple_of))
        model.validate_crop_size(crop_size)
        return model

    # crop_size_multiple == 1: Any crop size is allowed.
    m = create_model((9, 10, 11), 1)
    assert m.crop_size_constraints.multiple_of == (1, 1, 1)
    m = create_model((10, 12, 14), (5, 6, 7))
    assert m.crop_size_constraints.multiple_of == (5, 6, 7)

    def assert_fails(crop_size: TupleInt3, crop_size_multiple: IntOrTuple3) -> None:
        with pytest.raises(ValueError) as err:
            create_model(crop_size, crop_size_multiple)
        assert "Crop size is not valid" in str(err)

    # Crop size is not a multiple of 2 in dimensions 0 and 1
    assert_fails((3, 4, 5), 2)

    # Crop size constraints field should be generated as not empty if not provided in the
    # constructor - this simplifies code in the inference pipeline.
    model = SimpleModel(1, [1], 2, 2)
    assert model.crop_size_constraints is not None
    assert model.crop_size_constraints.minimum_size == (1, 1, 1)
    assert model.crop_size_constraints.multiple_of == (1, 1, 1)
    model.validate_crop_size((1, 1, 1))


def test_crop_size_multiple_in_build_net() -> None:
    """
    Tests if the the crop_size validation is really called in the model creation code
    """
    config = SegmentationModelBase(architecture=ModelArchitectureConfig.UNet3D,
                                   image_channels=["ct"],
                                   feature_channels=[1],
                                   kernel_size=3,
                                   crop_size=(17, 16, 16),
                                   should_validate=False)
    # Crop size of 17 in dimension 0 is invalid for a UNet3D, should be multiple of 16.
    # This should raise a ValueError.
    with pytest.raises(ValueError) as ex:
        build_net(config)
    assert "Training crop size: Crop size is not valid" in str(ex)
    config.crop_size = (16, 16, 16)
    config.test_crop_size = (17, 18, 19)
    with pytest.raises(ValueError) as ex:
        build_net(config)
    assert "Test crop size: Crop size is not valid" in str(ex)


def crop_size_multiple_unet3d(crop_size: TupleInt3,
                              downsampling_factor: IntOrTuple3,
                              num_downsampling_paths: int,
                              expected_crop_size_multiple: TupleInt3) -> None:
    model = UNet3D(name="",
                   input_image_channels=1,
                   initial_feature_channels=32,
                   num_classes=2,
                   kernel_size=3,
                   downsampling_factor=downsampling_factor,
                   num_downsampling_paths=num_downsampling_paths)
    assert model.crop_size_constraints.multiple_of == expected_crop_size_multiple
    model.validate_crop_size(crop_size)


def test_crop_size_multiple_unet3d() -> None:
    # Standard UNet with downsampling 2, 4 downsampling stages: Minimum crop is 16
    crop_size_multiple_unet3d(crop_size=(16, 64, 64), downsampling_factor=2, num_downsampling_paths=4,
                              expected_crop_size_multiple=(16, 16, 16))
    # Only 3 downsampling stages: each reduces by a factor 2, hence minimum crop size should be 8
    crop_size_multiple_unet3d(crop_size=(16, 64, 64), downsampling_factor=2, num_downsampling_paths=3,
                              expected_crop_size_multiple=(8, 8, 8))
    # UNet3D as used in UNet2D: No down-sampling in Z direction. Any crop size in Z is allowed, crop size
    # at least 16 in X and Y
    crop_size_multiple_unet3d(crop_size=(16, 64, 64), downsampling_factor=(1, 2, 2), num_downsampling_paths=4,
                              expected_crop_size_multiple=(1, 16, 16))
    # Non-standard downsampling factor of 3, 2 downsampling stages: Minimum crop is 3*3 = 9
    crop_size_multiple_unet3d(crop_size=(9, 9, 9), downsampling_factor=3, num_downsampling_paths=2,
                              expected_crop_size_multiple=(9, 9, 9))


def test_restrict_crop_size_too_small() -> None:
    """Test the modification of crop sizes when the image size is below the minimum."""
    shape = (10, 30, 40)
    crop_size = (20, 40, 20)
    stride = (10, 20, 20)
    constraint = CropSizeConstraints(multiple_of=16)
    with pytest.raises(ValueError) as e:
        constraint.restrict_crop_size_to_image(shape, crop_size, stride)
    # Error message should contain the actual image size
    assert str(shape) in e.value.args[0]
    assert "16" in e.value.args[0]


def check_restrict_crop(constraint: CropSizeConstraints,
                        shape: TupleInt3,
                        crop_size: TupleInt3,
                        stride: TupleInt3,
                        expected_crop: TupleInt3,
                        expected_stride: TupleInt3) -> None:
    (crop_new, stride_new) = constraint.restrict_crop_size_to_image(shape, crop_size, stride)
    assert crop_new == expected_crop
    assert stride_new == expected_stride
    # Stride and crop must be integer tuples
    assert isinstance(crop_new[0], int)
    assert isinstance(stride_new[0], int)


def test_restrict_crop_size() -> None:
    """Test the modification of crop sizes when the image size is smaller than the crop."""
    shape = (20, 35, 40)
    crop_size = (25, 40, 20)
    stride = (10, 20, 20)
    constraint = CropSizeConstraints(multiple_of=16)
    # Expected new crop size is the elementwise minimum of crop_size and shape,
    # rounded up to the nearest multiple of 16
    expected_crop = (32, 48, 32)
    # Stride should maintain (elementwise) the same ratio to crop as before
    expected_stride = (12, 24, 32)
    check_restrict_crop(constraint, shape, crop_size, stride, expected_crop, expected_stride)


def test_restrict_crop_size_tuple() -> None:
    """Test the modification of crop sizes when the image size is smaller than the crop,
    and the crop_multiple is a tuple with element-wise multiples."""
    shape = (20, 35, 40)
    crop_size = (25, 40, 20)
    stride = (10, 20, 20)
    constraint = CropSizeConstraints(multiple_of=(1, 16, 16))
    # Expected new crop size is the elementwise minimum of crop_size and shape,
    # rounded down to the nearest multiple of 16, apart from dimension 0
    expected_crop = (20, 48, 32)
    expected_stride = (8, 24, 32)
    check_restrict_crop(constraint, shape, crop_size, stride, expected_crop, expected_stride)


def test_restrict_crop_size_large_image() -> None:
    """Test the modification of crop sizes when the image size is larger than the crop:
    The crop and stride should be returned unchanged."""
    shape = (30, 50, 50)
    crop_size = (20, 40, 40)
    stride = crop_size
    constraint = CropSizeConstraints(multiple_of=1)
    expected_crop = crop_size
    expected_stride = stride
    check_restrict_crop(constraint, shape, crop_size, stride, expected_crop, expected_stride)
    constraint2 = CropSizeConstraints(multiple_of=1, minimum_size=999)
    with pytest.raises(ValueError) as err:
        check_restrict_crop(constraint2, shape, crop_size, stride, expected_crop, expected_stride)
    assert "at least a size of" in str(err)
    assert "999" in str(err)


def test_restrict_crop_size_uneven() -> None:
    """
    Test a case when the image is larger than the crop in Z, but not in X and Y.
    """
    shape = (20, 30, 30)
    crop_size = (10, 60, 60)
    stride = crop_size
    constraint = CropSizeConstraints(multiple_of=1)
    expected_crop = (10, 30, 30)
    expected_stride = (10, 30, 30)
    check_restrict_crop(constraint, shape, crop_size, stride, expected_crop, expected_stride)

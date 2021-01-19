#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import pytest

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_models import SegmentationLightning, create_lightning_model, \
    load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.pipelines.inference import InferencePipeline
from InnerEye.ML.utils import image_util
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


def run_inference_on_unet(size: TupleInt3) -> None:
    """
    Runs a model forward pass on a freshly created model, with an input image of the given size.
    Asserts that the model prediction has the same size as the input image.
    """
    fg_classes = ["tumour_mass", "subtract"]
    number_of_classes = len(fg_classes) + 1
    config = SegmentationModelBase(
        architecture="UNet3D",
        local_dataset=Path("dummy"),
        feature_channels=[1],
        kernel_size=3,
        largest_connected_component_foreground_classes=fg_classes,
        posterior_smoothing_mm=(2, 2, 2),
        crop_size=(64, 64, 64),
        # test_crop_size must be larger than 'size for the bug to trigger
        test_crop_size=(80, 80, 80),
        image_channels=["mr"],
        ground_truth_ids=fg_classes,
        ground_truth_ids_display_names=fg_classes,
        colours=[(255, 0, 0)] * len(fg_classes),
        fill_holes=[False] * len(fg_classes),
        mask_id=None,
        class_weights=[1.0 / number_of_classes] * number_of_classes,
        train_batch_size=8,
        inference_batch_size=1,
        inference_stride_size=(40, 40, 40),
        use_mixed_precision=True
    )
    lightning_model = create_lightning_model(config)
    assert isinstance(lightning_model, SegmentationLightning)
    pipeline = InferencePipeline(model=lightning_model, model_config=config)
    image = np.random.uniform(-1, 1, (1,) + size)
    result = pipeline.predict_and_post_process_whole_image(image, mask=np.ones(size), voxel_spacing_mm=(1, 1, 1))
    # All posteriors and segmentations must have the size of the input image
    for p in [*result.posteriors, result.segmentation]:
        assert p.shape == size
        # Check that all results are not NaN. In particular, if stride size is not adjusted
        # correctly, the results would be partially NaN.
        image_util.check_array_range(p)


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_inference_on_too_small_image() -> None:
    """
    Running inference on a simplified Unet model when the input image is too small along an axis.
    """
    with pytest.raises(ValueError) as ex:
        run_inference_on_unet((5, 10, 64))
    assert "input image must have at least a size of (16, 16, 16)" in str(ex)


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("size", [(26, 20, 50), (16, 16, 16)])
def test_inference_on_small_image(size: TupleInt3) -> None:
    """
    Test case for a failure at test time: Inference failed when
    the image was smaller than the test_crop_size. Try with different size, one that has
    multiples of 16 along all axis, one that has not.
    """
    run_inference_on_unet(size)


def test_invalid_stride_size(test_output_dirs: OutputFolderForTests) -> None:
    config = SegmentationModelBase(
        architecture="UNet3D",
        feature_channels=[1],
        crop_size=(64, 64, 64),
        test_crop_size=(80, 80, 80),
        image_channels=["mr"],
        ground_truth_ids=["tumour_mass", "subtract"],
        train_batch_size=8,
        inference_batch_size=1,
        inference_stride_size=(120, 120, 120),
        should_validate=False
    )
    config.set_output_to(test_output_dirs.root_dir)
    checkpoint_path = test_output_dirs.root_dir / "checkpoint.ckpt"
    create_model_and_store_checkpoint(config, checkpoint_path)

    with pytest.raises(ValueError) as ex:
        load_from_checkpoint_and_adjust_for_inference(config=config, checkpoint_path=checkpoint_path)

    assert "The inference stride size (120, 120, 120) must be smaller" in ex.value.args[0]
    assert str(config.inference_stride_size) in ex.value.args[0]
    assert str(config.test_crop_size) in ex.value.args[0]

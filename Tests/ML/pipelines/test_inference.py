#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import numpy as np
import pytest
import torch
from torch.nn import Parameter

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import InferencePipeline
from InnerEye.ML.utils import image_util
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("image_size", [(4, 4, 4), (4, 6, 8)])
@pytest.mark.parametrize("crop_size", [(5, 5, 5), (3, 3, 3), (3, 5, 7)])
def test_inference_image_and_crop_size(image_size: Any,
                                       crop_size: Any,
                                       test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(image_size=image_size,
                       crop_size=crop_size,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("shrink_by", [(0, 0, 0), (1, 1, 1), (1, 0, 1)])
def test_inference_shrink_y(shrink_by: Any,
                            test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(shrink_by=shrink_by,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("num_classes", [1, 5])
def test_inference_num_classes(num_classes: int,
                               test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(num_classes=num_classes,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("create_mask", [True, False])
def test_inference_create_mask(create_mask: bool,
                               test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(create_mask=create_mask,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("extract_largest_foreground_connected_component", [True, False])
def test_inference_component(extract_largest_foreground_connected_component: bool,
                             test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(extract_largest_foreground_connected_component=extract_largest_foreground_connected_component,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("is_ensemble", [True, False])
def test_inference_ensemble(is_ensemble: bool,
                            test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(is_ensemble=is_ensemble,
                       test_output_dirs=test_output_dirs)


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize("posterior_smoothing_mm", [None, (0.05, 0.06, 0.1)])
def test_inference_smoothing(posterior_smoothing_mm: Any,
                             test_output_dirs: OutputFolderForTests) -> None:
    inference_identity(posterior_smoothing_mm=posterior_smoothing_mm,
                       test_output_dirs=test_output_dirs)


class InferenceIdentityModel(SegmentationModelBase):
    def __init__(self, shrink_by: Any) -> None:
        super().__init__(should_validate=False)
        self.shrink_by = shrink_by

    def create_model(self) -> torch.nn.Module:
        return PyTorchMockModel(self.shrink_by)


def inference_identity(test_output_dirs: OutputFolderForTests,
                       image_size: Any = (4, 5, 8),
                       crop_size: Any = (5, 5, 5),
                       shrink_by: Any = (0, 0, 0),
                       num_classes: int = 5,
                       create_mask: bool = True,
                       extract_largest_foreground_connected_component: bool = False,
                       is_ensemble: bool = False,
                       posterior_smoothing_mm: Any = None) -> None:
    """
    Test to make sure inference pipeline is identity preserving, ie: we can recreate deterministic
    model output, ensuring the patching and stitching is robust.
    """
    # fix random seed
    np.random.seed(0)

    ground_truth_ids = list(map(str, range(num_classes)))
    # image to run inference on: The mock model passes the input image through, hence the input
    # image must have as many channels as we have classes (plus background), such that the output is
    # also a valid posterior.
    num_channels = num_classes + 1
    image_channels = np.random.randn(num_channels, *list(image_size))
    # create a random mask if required
    mask = np.round(np.random.uniform(size=image_size)).astype(np.int) if create_mask else None
    config = InferenceIdentityModel(shrink_by=shrink_by)
    config.crop_size = crop_size
    config.test_crop_size = crop_size
    config.image_channels = list(map(str, range(num_channels)))
    config.ground_truth_ids = ground_truth_ids
    config.posterior_smoothing_mm = posterior_smoothing_mm

    # We have to set largest_connected_component_foreground_classes after creating the model config,
    # because this parameter is not overridable and hence will not be set by GenericConfig's constructor.
    if extract_largest_foreground_connected_component:
        config.largest_connected_component_foreground_classes = [(c, None) for c in ground_truth_ids]
    # set expected posteriors
    expected_posteriors = torch.nn.functional.softmax(torch.tensor(image_channels), dim=0).numpy()
    # apply the mask if required
    if mask is not None:
        expected_posteriors = image_util.apply_mask_to_posteriors(expected_posteriors, mask)
    if posterior_smoothing_mm is not None:
        expected_posteriors = image_util.gaussian_smooth_posteriors(
            posteriors=expected_posteriors,
            kernel_size_mm=posterior_smoothing_mm,
            voxel_spacing_mm=(1, 1, 1)
        )
    # compute expected segmentation
    expected_segmentation = image_util.posteriors_to_segmentation(expected_posteriors)
    if extract_largest_foreground_connected_component:
        largest_component = image_util.extract_largest_foreground_connected_component(
            multi_label_array=expected_segmentation)
        # make sure the test data is accurate by checking if more than one component exists
        assert not np.array_equal(largest_component, expected_segmentation)
        expected_segmentation = largest_component

    # instantiate the model
    checkpoint = test_output_dirs.root_dir / "checkpoint.ckpt"
    create_model_and_store_checkpoint(config, checkpoint_path=checkpoint)

    # create single or ensemble inference pipeline
    inference_pipeline = InferencePipeline.create_from_checkpoint(path_to_checkpoint=checkpoint,
                                                                  model_config=config)
    assert inference_pipeline is not None
    full_image_inference_pipeline = EnsemblePipeline([inference_pipeline], config) \
        if is_ensemble else inference_pipeline

    # compute full image inference results
    inference_result = full_image_inference_pipeline \
        .predict_and_post_process_whole_image(image_channels=image_channels, mask=mask, voxel_spacing_mm=(1, 1, 1))

    # Segmentation must have same size as input image
    assert inference_result.segmentation.shape == image_size
    assert inference_result.posteriors.shape == (num_classes + 1,) + image_size
    # check that the posteriors and segmentations are as expected. Flatten to a list so that the error
    # messages are more informative.
    assert np.allclose(inference_result.posteriors, expected_posteriors)
    assert np.array_equal(inference_result.segmentation, expected_segmentation)


class PyTorchMockModel(BaseSegmentationModel):
    """
    Defines a model that returns a center crop of its input tensor. The center crop is defined by
    shrinking the image dimensions by a given amount, on either size of each axis.
    For example, if shrink_by is (0,1,5), the center crop is the input size in the first dimension unchanged,
    reduced by 2 in the second dimension, and reduced by 10 in the third.
    """

    def __init__(self, shrink_by: TupleInt3):
        super().__init__(input_channels=1, name='MockModel')
        # Create a fake parameter so that we can instantiate an optimizer easily
        self.foo = Parameter(requires_grad=True)
        self.shrink_by = shrink_by

    def forward(self, patches: np.ndarray) -> torch.Tensor:  # type: ignore
        # simulate models where only the center of the patch is returned
        image_shape = patches.shape[2:]

        def shrink_dim(i: int) -> int:
            return image_shape[i] - 2 * self.shrink_by[i]

        output_size = (shrink_dim(0), shrink_dim(1), shrink_dim(2))
        predictions = torch.zeros(patches.shape[:2] + output_size)
        for i, patch in enumerate(patches):
            for j, channel in enumerate(patch):
                predictions[i, j] = image_util.get_center_crop(image=channel, crop_shape=output_size)

        return predictions

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.ML.metrics_dict import MetricsDict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
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
from Tests.ML.configs.DummyModel import DummyModel
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.dataset.sample import PatientMetadata, Sample
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.model_testing import store_inference_results, evaluate_model_predictions, populate_metrics_writer


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


def create_config_from_dataset(input_list: List[List[str]], train: List[str], val: List[str], test: List[str]) \
        -> DummyModel:
    """
    Creates an "DummyModel(SegmentationModelBase)" object given patient list
    and training, validation and test subjects id.
    """

    class MyDummyModel(DummyModel):
        def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
            return DatasetSplits(train=dataset_df[dataset_df.subject.isin(train)],
                                 test=dataset_df[dataset_df.subject.isin(test)],
                                 val=dataset_df[dataset_df.subject.isin(val)])

    config = MyDummyModel()
    # Sets two regions for ground truth
    config.fg_ids = ["region", "region_1"]
    config.ground_truth_ids = config.fg_ids
    config.ground_truth_ids_display_names = config.fg_ids
    config.colours = [(255, 255, 255)] * len(config.fg_ids)
    config.fill_holes = [False] * len(config.fg_ids)
    config.roi_interpreted_types = ["Organ"] * len(config.fg_ids)
    config.check_exclusive = False
    df = pd.DataFrame(input_list, columns=['subject', 'filePath', 'channel'])
    config._dataset_data_frame = df
    return config


def test_evaluate_model_predictions() -> None:
    """
    Creates an 'InferencePipeline.Result' object using pre-defined volumes, stores results and evaluates metrics.
    """
    # Patient 3,4,5 are in test dataset such that:
    # Patient 3 has one missing ground truth channel: "region"
    # Patient 4 has all missing ground truth channels: "region", "region_1"
    # Patient 5 has no missing ground truth channels.
    input_list = [
        ["1", "train_and_test_data/id1_channel1.nii.gz", "channel1"],
        ["1", "train_and_test_data/id1_channel1.nii.gz", "channel2"],
        ["1", "train_and_test_data/id1_mask.nii.gz", "mask"],
        ["1", "train_and_test_data/id1_region.nii.gz", "region"],
        ["1", "train_and_test_data/id1_region.nii.gz", "region_1"],
        ["2", "train_and_test_data/id2_channel1.nii.gz", "channel1"],
        ["2", "train_and_test_data/id2_channel1.nii.gz", "channel2"],
        ["2", "train_and_test_data/id2_mask.nii.gz", "mask"],
        ["2", "train_and_test_data/id2_region.nii.gz", "region"],
        ["2", "train_and_test_data/id2_region.nii.gz", "region_1"],
        ["3", "train_and_test_data/id2_channel1.nii.gz", "channel1"],
        ["3", "train_and_test_data/id2_channel1.nii.gz", "channel2"],
        ["3", "train_and_test_data/id2_mask.nii.gz", "mask"],
        # ["3", "train_and_test_data/id2_region.nii.gz", "region"], # commented on purpose
        ["3", "train_and_test_data/id2_region.nii.gz", "region_1"],
        ["4", "train_and_test_data/id2_channel1.nii.gz", "channel1"],
        ["4", "train_and_test_data/id2_channel1.nii.gz", "channel2"],
        ["4", "train_and_test_data/id2_mask.nii.gz", "mask"],
        # ["4", "train_and_test_data/id2_region.nii.gz", "region"], # commented on purpose
        # ["4", "train_and_test_data/id2_region.nii.gz", "region_1"], # commented on purpose
        ["5", "train_and_test_data/id2_channel1.nii.gz", "channel1"],
        ["5", "train_and_test_data/id2_channel1.nii.gz", "channel2"],
        ["5", "train_and_test_data/id2_mask.nii.gz", "mask"],
        ["5", "train_and_test_data/id2_region.nii.gz", "region"],
        ["5", "train_and_test_data/id2_region.nii.gz", "region_1"]]

    config = create_config_from_dataset(input_list, train=['1'], val=['2'], test=['3', '4', '5'])
    ds = config.get_torch_dataset_for_inference(ModelExecutionMode.TEST)
    results_folder = config.outputs_folder
    if not results_folder.is_dir():
        results_folder.mkdir()

    model_prediction_evaluations: List[Tuple[PatientMetadata, MetricsDict]] = []

    for sample_index, sample in enumerate(ds, 1):
        sample = Sample.from_dict(sample=sample)
        posteriors = np.zeros((3,) + sample.mask.shape, 'float32')
        posteriors[0][:] = 0.2
        posteriors[1][:] = 0.6
        posteriors[2][:] = 0.2

        assert config.dataset_expected_spacing_xyz is not None

        inference_result = InferencePipeline.Result(
            patient_id=sample.patient_id,
            posteriors=posteriors,
            segmentation=np.argmax(posteriors, 0),
            voxel_spacing_mm=config.dataset_expected_spacing_xyz
        )
        store_inference_results(inference_result=inference_result,
                                config=config,
                                results_folder=results_folder,
                                image_header=sample.metadata.image_header)

        metadata, metrics_per_class = evaluate_model_predictions(
            sample_index - 1,
            config=config,
            dataset=ds,
            results_folder=results_folder)

        model_prediction_evaluations.append((metadata, metrics_per_class))

        # Patient 3 has one missing ground truth channel: "region"
        if sample.metadata.patient_id == '3':
            assert 'Dice' in metrics_per_class.values('region_1').keys()
            assert 'HausdorffDistance_millimeters' in metrics_per_class.values('region_1').keys()
            assert 'MeanSurfaceDistance_millimeters' in metrics_per_class.values('region_1').keys()
            for hue_name in ['region', 'Default']:
                for metric_type in metrics_per_class.values(hue_name).keys():
                    assert np.isnan(metrics_per_class.values(hue_name)[metric_type]).all()

        # Patient 4 has all missing ground truth channels: "region", "region_1"
        if sample.metadata.patient_id == '4':
            for hue_name in ['region_1', 'region', 'Default']:
                for metric_type in metrics_per_class.values(hue_name).keys():
                    assert np.isnan(metrics_per_class.values(hue_name)[metric_type]).all()

        # Patient 5 has no missing ground truth channels
        if sample.metadata.patient_id == '5':
            for metric_type in metrics_per_class.values('Default').keys():
                assert np.isnan(metrics_per_class.values('Default')[metric_type]).all()
            for hue_name in ['region_1', 'region']:
                assert 'Dice' in metrics_per_class.values(hue_name).keys()
                assert 'HausdorffDistance_millimeters' in metrics_per_class.values(hue_name).keys()
                assert 'MeanSurfaceDistance_millimeters' in metrics_per_class.values(hue_name).keys()

    metrics_writer, average_dice = populate_metrics_writer(model_prediction_evaluations, config)
    # Patient 3 has only one missing ground truth channel
    assert not np.isnan(average_dice[0])
    assert np.isnan(float(metrics_writer.columns["Dice"][0]))
    assert not np.isnan(float(metrics_writer.columns["Dice"][1]))
    assert np.isnan(float(metrics_writer.columns["HausdorffDistance_mm"][0]))
    assert not np.isnan(float(metrics_writer.columns["HausdorffDistance_mm"][1]))
    assert np.isnan(float(metrics_writer.columns["MeanDistance_mm"][0]))
    assert not np.isnan(float(metrics_writer.columns["MeanDistance_mm"][1]))
    # Patient 4 has all missing ground truth channels
    assert np.isnan(average_dice[1])
    assert np.isnan(float(metrics_writer.columns["Dice"][2]))
    assert np.isnan(float(metrics_writer.columns["Dice"][3]))
    assert np.isnan(float(metrics_writer.columns["HausdorffDistance_mm"][2]))
    assert np.isnan(float(metrics_writer.columns["HausdorffDistance_mm"][3]))
    assert np.isnan(float(metrics_writer.columns["MeanDistance_mm"][2]))
    assert np.isnan(float(metrics_writer.columns["MeanDistance_mm"][3]))
    # Patient 5 has no missing ground truth channels.
    assert average_dice[2] > 0
    assert float(metrics_writer.columns["Dice"][4]) >= 0
    assert float(metrics_writer.columns["Dice"][5]) >= 0
    assert float(metrics_writer.columns["HausdorffDistance_mm"][4]) >= 0
    assert float(metrics_writer.columns["HausdorffDistance_mm"][5]) >= 0
    assert float(metrics_writer.columns["MeanDistance_mm"][4]) >= 0
    assert float(metrics_writer.columns["MeanDistance_mm"][5]) >= 0

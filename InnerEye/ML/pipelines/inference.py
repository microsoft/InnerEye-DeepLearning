#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torchio as tio

from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML import config
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_helpers import load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.lightning_models import SegmentationLightning
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils import image_util, ml_util
from InnerEye.ML.utils.image_util import compute_uncertainty_map_from_posteriors, gaussian_smooth_posteriors, \
    posteriors_to_segmentation


class InferencePipelineBase:
    """Base class for all inference pipelines."""

    def __init__(self, model_config: ModelConfigBase):
        self.model_config = model_config


class FullImageInferencePipelineBase(InferencePipelineBase):
    """
    Base Class for full image inference intended to be inherited by inference pipelines
    that can perform full image prediction
    """

    def __init__(self, model_config: SegmentationModelBase):
        super().__init__(model_config)

    def predict_and_post_process_whole_image(self, image_channels: np.ndarray,
                                             voxel_spacing_mm: TupleFloat3,
                                             mask: np.ndarray = None,
                                             patient_id: int = 0) -> InferencePipeline.Result:
        return self.post_process(self.predict_whole_image(image_channels, voxel_spacing_mm, mask, patient_id))

    def predict_whole_image(self, image_channels: np.ndarray,
                            voxel_spacing_mm: TupleFloat3,
                            mask: np.ndarray = None,
                            patient_id: int = 0) -> InferencePipeline.Result:
        raise NotImplementedError("Full image inference capability must be implemented by concrete classes")

    def post_process(self, results: InferencePipeline.Result) -> InferencePipeline.Result:
        """
        Perform connected component analysis to update segmentation with largest
        connected component based on the configurations
        :param results: inference results to post-process
        :return: post-processed version of results
        """
        if self.model_config.posterior_smoothing_mm:
            posteriors = gaussian_smooth_posteriors(
                posteriors=results.posteriors,
                kernel_size_mm=self.model_config.posterior_smoothing_mm,
                voxel_spacing_mm=results.voxel_spacing_mm
            )

            results = InferencePipeline.Result(
                patient_id=results.patient_id,
                posteriors=posteriors,
                segmentation=posteriors_to_segmentation(posteriors),
                voxel_spacing_mm=results.voxel_spacing_mm
            )

        if self.model_config.summed_probability_rules and not self.model_config.disable_extra_postprocessing:
            assert isinstance(self.model_config, SegmentationModelBase)
            results = results.with_new_segmentation(
                image_util.apply_summed_probability_rules(self.model_config, results.posteriors, results.segmentation))

        if self.model_config.largest_connected_component_foreground_classes is not None:
            # get indices for classes to restrict
            restrict_class_indices_and_thresholds = []
            for name, idx in self.model_config.class_and_index_with_background().items():
                for name2, threshold in self.model_config.largest_connected_component_foreground_classes:
                    if name2 == name:
                        restrict_class_indices_and_thresholds.append((idx, threshold))
            results = results.with_new_segmentation(
                image_util.extract_largest_foreground_connected_component(
                    multi_label_array=results.segmentation,
                    # mypy gets confused below because List is invariant. Sequence is covariant
                    # but does not allow "append".
                    restrictions=restrict_class_indices_and_thresholds))  # type: ignore

        if self.model_config.slice_exclusion_rules and not self.model_config.disable_extra_postprocessing:
            results = results.with_new_segmentation(
                image_util.apply_slice_exclusion_rules(self.model_config, results.segmentation))

        return results


class InferencePipeline(FullImageInferencePipelineBase):
    """
    Pipeline class for model for whole image inference on ct-images.
    """

    # the model output is expected to be a valid probability distribution
    MODEL_OUTPUT_POSTERIOR_RANGE = (0, 1)

    class Variables(Enum):
        """
        Variables associated with the inference pipeline
        """

        # an instantiated model to use for inference.
        Model = 'model'
        # the configuration associated with the model.
        ModelConfig = 'model_config'
        # the shape of the image required as output from the pipeline.
        OutputImageShape = 'output_image_shape'
        # A Tuple[int,int,int] with the crop size that should be used. For large images, this will be
        # the test_crop_size from the model config, but for smaller images, it will be the componentwise
        # minimum of test_crop_size and image_size
        CropSize = 'crop_size'
        # The stride size to use, possibly adjusted for small images (see above for crop_size)
        Stride = 'stride'
        # The size of the output tensor that the model will produce when fed with an input tensor that
        # has the given crop_size.
        OutputSize = 'output_size'

    class Result:
        """
        Contains the inference results from a single pass of the inference pipeline
        """

        def __init__(self,
                     patient_id: int,
                     segmentation: np.ndarray,
                     posteriors: np.ndarray,
                     voxel_spacing_mm: TupleFloat3):
            """
            :param patient_id: The id of the patient instance for with inference is being performed on.
            :param segmentation: Z x Y x X (argmaxed over the posteriors in the class dimension)
            :param voxel_spacing_mm: Voxel spacing to use for each dimension in (Z x Y x X) order
            :param posteriors: Class x Z x Y x X
            """
            self.patient_id = patient_id
            self.segmentation = segmentation
            self.posteriors = posteriors
            self.voxel_spacing_mm = voxel_spacing_mm

            if len(self.voxel_spacing_mm) != 3:
                raise ValueError(f"voxel_spacing_mm must have length 3, found: {voxel_spacing_mm}")
            if any(np.array(self.voxel_spacing_mm) <= 0):
                raise ValueError(f"voxel_spacing_mm must have values > 0 in each dimension, found: {voxel_spacing_mm}")

            ml_util.check_size_matches(self.segmentation,
                                       self.posteriors,
                                       dim1=3,
                                       dim2=4,
                                       matching_dimensions=[-3, -2, -1],
                                       arg1_name="segmentation",
                                       arg2_name="posteriors")

            segmentation_value_range = np.unique(self.segmentation)
            if not np.all([x in range(self.posteriors.shape[0]) for x in segmentation_value_range]):
                raise Exception("values in the segmentation map must be in range [0, classes), "
                                "found classes:{}, segmentation range:{}"
                                .format(self.posteriors.shape[0], segmentation_value_range))

            self._uncertainty = compute_uncertainty_map_from_posteriors(self.posteriors)

        @property
        def uncertainty(self) -> np.ndarray:
            return self._uncertainty

        def with_new_segmentation(self, segmentation: np.ndarray) -> InferencePipeline.Result:
            if segmentation.shape != self.segmentation.shape:
                raise ValueError(f"Attempt to replace segmentation of shape {self.segmentation.shape} "
                                 f"with one of shape {segmentation.shape}")
            return InferencePipeline.Result(
                patient_id=self.patient_id,
                segmentation=segmentation,
                posteriors=self.posteriors,
                voxel_spacing_mm=self.voxel_spacing_mm)

    def __init__(self, model: SegmentationLightning, model_config: config.SegmentationModelBase,
                 pipeline_id: int = 0):
        super().__init__(model_config)
        self.model = model
        self.model.model.eval()
        self.pipeline_id = pipeline_id

    @staticmethod
    def create_from_checkpoint(path_to_checkpoint: Path,
                               model_config: SegmentationModelBase,
                               pipeline_id: int = 0) -> Optional[InferencePipeline]:
        """
        Creates an instance of the inference pipeline for a given epoch from a stored checkpoint.
        After loading, the model parameters are checked for NaN and Infinity values.
        If there is no checkpoint file for the given epoch, return None.
        :param path_to_checkpoint: The path to the checkpoint that we want to load
        model_config.checkpoint_folder
        :param model_config: Model related configurations.
        :param pipeline_id: Numeric identifier for the pipeline (useful for logging when ensembling)
        :return InferencePipeline: an instantiated inference pipeline instance, or None if there was no checkpoint
        file for this epoch.
        """
        if not path_to_checkpoint.is_file():
            # not raising a value error here: This is used to create individual pipelines for ensembles,
            #                                   possible one model cannot be created but others can
            logging.warning(f"Could not recover model from checkpoint path {path_to_checkpoint}")
            return None
        lightning_model = load_from_checkpoint_and_adjust_for_inference(model_config, path_to_checkpoint)
        assert isinstance(lightning_model, SegmentationLightning)
        return InferencePipeline(model=lightning_model, model_config=model_config, pipeline_id=pipeline_id)

    def post_process_posteriors(self, posteriors: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform post processing on the computed outputs of the a single pass of the pipelines.
        Currently the following operations are performed:
        -------------------------------------------------------------------------------------
        1) the mask is applied to the posteriors (if required).
        2) the final posteriors are used to perform an argmax to generate a multi-label segmentation.
        3) extract the largest foreground connected component in the segmentation if required
        """
        if mask is not None:
            posteriors = image_util.apply_mask_to_posteriors(posteriors=posteriors, mask=mask)

        # create segmentation using an argmax over the posterior probabilities
        segmentation = image_util.posteriors_to_segmentation(posteriors)

        return posteriors, segmentation

    @torch.no_grad()
    def predict_whole_image(self, image_channels: np.ndarray,
                            voxel_spacing_mm: TupleFloat3,
                            mask: Optional[np.ndarray] = None,
                            patient_id: int = 0) -> InferencePipeline.Result:
        """
        Performs a single inference pass through the pipeline for the provided image
        :param image_channels: The input image channels to perform inference on in format: Channels x Z x Y x X.
        :param voxel_spacing_mm: Voxel spacing to use for each dimension in (Z x Y x X) order
        :param mask: A binary image used to ignore results outside it in format: Z x Y x X.
        :param patient_id: The identifier of the patient this image belongs to (defaults to 0 if None provided).
        :return InferenceResult: that contains Segmentation for each of the classes and their posterior probabilities.
        """
        if image_channels is None:
            raise Exception("image_channels cannot be None")
        if image_channels.ndim != 4:
            raise NotImplementedError("image_channels must be in shape: Channels x Z x Y x X"
                                      "found image_channels shape: {}".format(image_channels.shape))
        if mask is not None:
            ml_util.check_size_matches(image_channels, mask, 4, 3, [-1, -2, -3])
        self.model.eval()

        image = tio.ScalarImage(tensor=image_channels)
        INPUT = 'input_image'
        MASK = 'mask'

        subject_dict: Dict[str, tio.Image] = {INPUT: image}
        if mask is not None:
            subject_dict[MASK] = tio.LabelMap(tensor=mask[np.newaxis])
        subject = tio.Subject(subject_dict)

        constraints = self.model.model.crop_size_constraints

        # Make sure the image size is compatible with the model
        ensure_shape_multiple = tio.EnsureShapeMultiple(constraints.multiple_of)
        subject = ensure_shape_multiple(subject)  # type: ignore

        # There may be cases where the test image is smaller than the test_crop_size. Adjust crop_size
        # to always fit into image. If test_crop_size is smaller than the image, crop will remain unchanged.
        restrict_patch_size = constraints.restrict_crop_size_to_image  # type: ignore
        effective_patch_size, effective_stride = restrict_patch_size(subject.spatial_shape,  # type: ignore
                                                                    self.model_config.test_crop_size,
                                                                    self.model_config.inference_stride_size)

        patch_overlap = np.array(effective_patch_size) - np.array(effective_stride)
        grid_sampler = tio.inference.GridSampler(
            subject,
            effective_patch_size,
            patch_overlap,
            padding_mode=self.model_config.padding_mode.value,
        )
        batch_size = self.model_config.inference_batch_size
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)  # type: ignore
        aggregator = tio.inference.GridAggregator(grid_sampler)

        logging.debug(
            f"Inference on image size {subject.spatial_shape} will run "
            f"with crop size {effective_patch_size} and stride {effective_stride}")
        for patches_batch in patch_loader:
            input_tensor = patches_batch[INPUT][tio.DATA].float()
            if self.model_config.use_gpu:
                input_tensor = input_tensor.cuda()
            locations = patches_batch[tio.LOCATION]
            # perform the forward pass
            patches_posteriors = self.model(input_tensor).detach()
            # collect the predictions over each of the batches
            aggregator.add_batch(patches_posteriors, locations)
        posteriors = aggregator.get_output_tensor().numpy()
        posteriors_mask = None if mask is None else subject[MASK].numpy()[0]
        posteriors, segmentation = self.post_process_posteriors(posteriors, mask=posteriors_mask)

        image_util.check_array_range(posteriors, error_prefix="Whole image posteriors")

        # Make sure the final shape matches the input shape by undoing the padding in EnsureShapeMultiple (if any)
        posteriors_image = tio.ScalarImage(tensor=posteriors, affine=image.affine)
        segmentation_image = tio.LabelMap(tensor=segmentation[np.newaxis], affine=image.affine)
        subject.add_image(posteriors_image, 'posteriors')
        subject.add_image(segmentation_image, 'segmentation')
        # Remove some images to avoid unnecessary computations
        subject.remove_image(INPUT)
        if mask is not None:
            subject.remove_image(MASK)
        subject_original_space = subject.apply_inverse_transform()
        posteriors = subject_original_space.posteriors.numpy()  # type: ignore
        segmentation = subject_original_space.segmentation.numpy()[0]  # type: ignore

        # prepare pipeline results from the processed batch
        return InferencePipeline.Result(
            patient_id=patient_id,
            segmentation=segmentation,
            posteriors=posteriors,
            voxel_spacing_mm=voxel_spacing_mm
        )

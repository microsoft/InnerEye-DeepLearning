#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from radio import CTImagesMaskedBatch
from radio.batchflow import Dataset, action, inbatch_parallel

from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML import config
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.lightning_models import SegmentationLightning, load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.models.architectures.base_model import BaseSegmentationModel
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

    def predict_whole_image(self, image_channels: np.ndarray,
                            voxel_spacing_mm: TupleFloat3,
                            mask: np.ndarray = None,
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
        # create the dataset for the batch
        batch_dataset = Dataset(index=[patient_id], batch_class=InferenceBatch)
        # setup the pipeline
        pipeline = (batch_dataset.p
                    # define pipeline variables
                    .init_variables([InferencePipeline.Variables.Model,
                                     InferencePipeline.Variables.ModelConfig,
                                     InferencePipeline.Variables.CropSize,
                                     InferencePipeline.Variables.OutputSize,
                                     InferencePipeline.Variables.OutputImageShape,
                                     InferencePipeline.Variables.Stride])
                    # update the variables for the batch actions
                    .update_variable(name=InferencePipeline.Variables.Model, value=self.model)
                    .update_variable(name=InferencePipeline.Variables.ModelConfig, value=self.model_config)
                    # perform cascaded batch actions
                    .load(image_channels=image_channels, mask=mask)
                    .pre_process()
                    .predict()
                    .post_process()
                    )
        # run the batch through the pipeline
        logging.info(f"Inference pipeline ({self.pipeline_id}), Predicting patient: {patient_id}")
        processed_batch: InferenceBatch = pipeline.next_batch(batch_size=1)
        posteriors = processed_batch.get_component(InferenceBatch.Components.Posteriors)
        image_util.check_array_range(posteriors, error_prefix="Whole image posteriors")
        # prepare pipeline results from the processed batch
        return InferencePipeline.Result(
            patient_id=patient_id,
            segmentation=processed_batch.get_component(InferenceBatch.Components.Segmentation),
            posteriors=posteriors,
            voxel_spacing_mm=voxel_spacing_mm
        )


class InferenceBatch(CTImagesMaskedBatch):
    """
    Batch class for IO with the inference pipeline. One instance of a batch will load the image
    into the 'images' component of the pipeline, and store the results of the full pass
    of the pipeline into the 'segmentation' and 'posteriors' components.
    """

    class Components(Enum):
        """
        Components associated with the inference batch class
        """

        # the input image channels in Channels x Z x Y x X format.
        ImageChannels = 'channels'
        # a set of 2D image slices (ie: a 3D image channel), stacked in Z x Y x X format.
        Images = 'images'
        # a binary mask used to ignore predictions in Z x Y x X format.
        Mask = 'mask'
        # a numpy.ndarray in Z x Y x X format with class labels for each voxel in the original image.
        Segmentation = 'segmentation'
        # a numpy.ndarray with the first dimension indexing each class in C x Z x Y x X format
        # with each Z x Y x X being the same shape as the Images component, and consisting of
        # [0, 1] values representing the model confidence for each voxel.
        Posteriors = 'posteriors'

    def __init__(self, index: int, *args: Any, **kwargs: Any):
        super().__init__(index, *args, **kwargs)
        self.components = [x.value for x in InferenceBatch.Components]

    @action
    def load(self, image_channels: np.ndarray, mask: np.ndarray) -> InferenceBatch:
        """
        Load image channels and mask into their respective pipeline components.
        """
        self.set_component(component=InferenceBatch.Components.ImageChannels, data=image_channels)
        model_config = self.get_configs()
        if model_config is None:
            raise ValueError("model_config is None")
        if model_config.test_crop_size is None:
            raise ValueError("model_config.test_crop_size is None")
        if model_config.inference_stride_size is None:
            raise ValueError("model_config.inference_stride_size is None")

        # fetch the image channels from the batch
        image_channels = self.get_component(InferenceBatch.Components.ImageChannels)
        self.pipeline.set_variable(name=InferencePipeline.Variables.OutputImageShape, value=image_channels[0].shape)
        # There may be cases where the test image is smaller than the test_crop_size. Adjust crop_size
        # to always fit into image. If test_crop_size is smaller than the image, crop will remain unchanged.
        image_size = image_channels.shape[1:]
        model: BaseSegmentationModel = self.pipeline.get_variable(InferencePipeline.Variables.Model).model
        effective_crop, effective_stride = \
            model.crop_size_constraints.restrict_crop_size_to_image(image_size,
                                                                    model_config.test_crop_size,
                                                                    model_config.inference_stride_size)
        self.pipeline.set_variable(name=InferencePipeline.Variables.CropSize, value=effective_crop)
        self.pipeline.set_variable(name=InferencePipeline.Variables.Stride, value=effective_stride)
        logging.debug(
            f"Inference on image size {image_size} will run "
            f"with crop size {effective_crop} and stride {effective_stride}")
        # In most cases, we will be able to read the output size from the pre-computed values
        # via get_output_size. Only if we have a non-standard (smaller) crop size, re-computed the output size.
        output_size = model_config.get_output_size(execution_mode=ModelExecutionMode.TEST)
        if effective_crop != model_config.test_crop_size:
            output_size = model.get_output_shape(input_shape=effective_crop)  # type: ignore
        self.pipeline.set_variable(name=InferencePipeline.Variables.OutputSize, value=output_size)

        if mask is not None:
            self.set_component(component=InferenceBatch.Components.Mask, data=mask)

        return self

    @action
    def pre_process(self) -> InferenceBatch:
        """
        Prepare the input components of the batch for further processing.
        """
        model_config = self.get_configs()

        # fetch the image channels from the batch
        image_channels = self.get_component(InferenceBatch.Components.ImageChannels)

        crop_size = self.pipeline.get_variable(InferencePipeline.Variables.CropSize)
        output_size = self.pipeline.get_variable(InferencePipeline.Variables.OutputSize)
        image_channels = image_util.pad_images_for_inference(
            images=image_channels,
            crop_size=crop_size,
            output_size=output_size,
            padding_mode=model_config.padding_mode
        )

        # update the post-processed components
        self.set_component(component=InferenceBatch.Components.ImageChannels, data=image_channels)

        return self

    @action
    def predict(self) -> InferenceBatch:
        """
        Perform a forward pass of the model on the provided image, this generates
        a set of posterior maps for each class, as well as a segmentation output
        stored in the respective 'posteriors' and 'segmentation' components.
        """
        model_config = self.get_configs()

        # extract patches for each image channel: Num patches x Channels x Z x Y x X
        patches = self._extract_patches_for_image_channels()

        # split the generated patches into batches and perform forward passes
        predictions = []
        batch_size = model_config.inference_batch_size

        for batch_idx in range(0, len(patches), batch_size):
            # slice over the batches to prepare batch
            batch = torch.tensor(patches[batch_idx: batch_idx + batch_size, ...]).float()
            if model_config.use_gpu:
                batch = batch.cuda()
            # perform the forward pass
            batch_predictions = self._model_fn(batch).detach().cpu().numpy()
            # collect the predictions over each of the batches
            predictions.append(batch_predictions)

        # map the batched predictions to the original batch shape
        # of shape but with an added class dimension: Num patches x Class x Z x Y x X
        predictions = np.concatenate(predictions, axis=0)

        # create posterior output for each class with the shape: Class x Z x Y x x. We use float32 as these
        # arrays can be big.
        output_image_shape = self.pipeline.get_variable(InferencePipeline.Variables.OutputImageShape)
        posteriors = np.zeros(shape=[model_config.number_of_classes] + list(output_image_shape), dtype=np.float32)
        stride = self.pipeline.get_variable(InferencePipeline.Variables.Stride)

        for c in range(len(posteriors)):
            # stitch the patches for each posterior class
            self.load_from_patches(predictions[:, c, ...],  # type: ignore
                                   stride=stride,
                                   scan_shape=output_image_shape,
                                   data_attr=InferenceBatch.Components.Posteriors.value)
            # extract computed output from the component so the pipeline buffer can be reused
            posteriors[c] = self.get_component(InferenceBatch.Components.Posteriors)

        # store the stitched up results for the batch
        self.set_component(component=InferenceBatch.Components.Posteriors, data=posteriors)

        return self

    @action
    def post_process(self) -> InferenceBatch:
        """
        Perform post processing on the computed outputs of the a single pass of the pipelines.
        Currently the following operations are performed:
        -------------------------------------------------------------------------------------
        1) the mask is applied to the posteriors (if required).
        2) the final posteriors are used to perform an argmax to generate a multi-label segmentation.
        3) extract the largest foreground connected component in the segmentation if required
        """
        mask = self.get_component(InferenceBatch.Components.Mask)
        posteriors = self.get_component(InferenceBatch.Components.Posteriors)
        if mask is not None:
            posteriors = image_util.apply_mask_to_posteriors(posteriors=posteriors, mask=mask)

        # create segmentation using an argmax over the posterior probabilities
        segmentation = image_util.posteriors_to_segmentation(posteriors)

        # update the post-processed posteriors and save the segmentation
        self.set_component(component=InferenceBatch.Components.Posteriors, data=posteriors)
        self.set_component(component=InferenceBatch.Components.Segmentation, data=segmentation)

        return self

    def get_configs(self) -> config.SegmentationModelBase:
        return self.pipeline.get_variable(InferencePipeline.Variables.ModelConfig)

    def get_component(self, component: InferenceBatch.Components) -> np.ndarray:
        return getattr(self, component.value) if hasattr(self, component.value) else None

    @inbatch_parallel(init='indices', post='_post_custom_components', target='threads')
    def set_component(self, batch_idx: int, component: InferenceBatch.Components, data: np.ndarray) \
            -> Dict[InferenceBatch.Components, Any]:
        logging.debug("Updated data in pipeline component: {}, for batch: {}.".format(component.value, batch_idx))
        return {
            component.value: {'type': component.value, 'data': data}
        }

    def _extract_patches_for_image_channels(self) -> np.ndarray:
        """
        Extracts deterministically, patches from each image channel
        :return: Patches for each image channel in format: Num patches x Channels x Z x Y x X
        """
        model_config = self.get_configs()
        image_channels = self.get_component(InferenceBatch.Components.ImageChannels)
        # There may be cases where the test image is smaller than the test_crop_size. Adjust crop_size
        # to always fit into image, and adjust stride accordingly. If test_crop_size is smaller than the
        # image, crop and stride will remain unchanged.
        crop_size = self.pipeline.get_variable(InferencePipeline.Variables.CropSize)
        stride = self.pipeline.get_variable(InferencePipeline.Variables.Stride)
        patches = []
        for channel_index, channel in enumerate(image_channels):
            # set the current image channel component to process
            self.set_component(component=InferenceBatch.Components.Images, data=channel)
            channel_patches = self.get_patches(patch_shape=crop_size,
                                               stride=stride,
                                               padding=model_config.padding_mode.value,
                                               data_attr=InferenceBatch.Components.Images.value)
            logging.debug(
                f"Image channel {channel_index}: Tensor with extracted patches has size {channel_patches.shape}")
            patches.append(channel_patches)
        # reset the images component
        self.set_component(component=InferenceBatch.Components.Images, data=[])

        return np.stack(patches, axis=1)

    def _model_fn(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Wrapper function to handle the model forward pass
        :param patches: Image patches to be passed to the model in format Patches x Channels x Z x Y x X
        :return posteriors: Confidence maps [0,1] for each patch per class
        in format: Patches x Channels x Class x Z x Y x X
        """
        model = self.pipeline.get_variable(InferencePipeline.Variables.Model)
        # Model forward pass returns posteriors
        with torch.no_grad():
            return model(patches)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from InnerEye.Common.type_annotations import TupleFloat3
from InnerEye.ML.config import EnsembleAggregationType, SegmentationModelBase
from InnerEye.ML.pipelines.inference import FullImageInferencePipelineBase, InferencePipeline
from InnerEye.ML.utils.image_util import posteriors_to_segmentation


class EnsemblePipeline(FullImageInferencePipelineBase):
    """
    Pipeline for ensembling model predictions for whole image inference
    """

    def __init__(self, inference_pipelines: List[InferencePipeline], model_config: SegmentationModelBase):
        super().__init__(model_config)
        self._inference_pipelines = inference_pipelines

        # check that all of the inference pipelines are not None
        if None in self._inference_pipelines:
            raise ValueError(f"All inference pipelines in the ensemble must be non None, "
                             f"found: {self._inference_pipelines}")

    @staticmethod
    def create_from_checkpoints(path_to_checkpoints: List[Path],
                                model_config: SegmentationModelBase) -> EnsemblePipeline:
        pipelines = []
        for i, path in enumerate(path_to_checkpoints):
            pipeline = InferencePipeline.create_from_checkpoint(path, model_config, i)
            if pipeline is None:
                logging.warning(f"Cannot create pipeline from path {path}; dropping it from ensemble")
            else:
                pipelines.append(pipeline)
        if not pipelines:
            raise ValueError("Could not create ANY pipelines from checkpoint paths")
        return EnsemblePipeline(model_config=model_config, inference_pipelines=pipelines)

    @staticmethod
    def aggregate_results(results: List[InferencePipeline.Result],
                          aggregation_type: EnsembleAggregationType) -> InferencePipeline.Result:
        """
        Helper method to aggregate results from multiple inference pipelines, based on the aggregation type provided.
        :param results: inference pipeline results to aggregate.
        :param aggregation_type: aggregation function to use to combine the results.
        :return: InferenceResult: contains a Segmentation for each of the classes and their posterior
        probabilities.
        """
        if aggregation_type != EnsembleAggregationType.Average:
            raise NotImplementedError(f"Ensembling is not implemented for aggregation type: {aggregation_type}")
        posteriors = np.mean([x.posteriors for x in results], axis=0)
        return InferencePipeline.Result(
            epoch=results[0].epoch,
            patient_id=results[0].patient_id,
            posteriors=posteriors,
            segmentation=posteriors_to_segmentation(posteriors),
            voxel_spacing_mm=results[0].voxel_spacing_mm)

    def predict_whole_image(self, image_channels: np.ndarray,
                            voxel_spacing_mm: TupleFloat3,
                            mask: np.ndarray = None,
                            patient_id: int = 0) -> InferencePipeline.Result:
        """
        Performs a single inference pass for each model in the ensemble, and aggregates the results
        based on the provided aggregation type.
        :param image_channels: The input image channels to perform inference on in format: Channels x Z x Y x X.
        :param voxel_spacing_mm: Voxel spacing to use for each dimension in (Z x Y x X) order
        :param mask: A binary image used to ignore results outside it in format: Z x Y x X.
        :param patient_id: The identifier of the patient this image belongs to.
        :return InferenceResult: that contains Segmentation for each of the classes and their posterior
        probabilities.
        """
        logging.info(f"Ensembling inference pipelines ({self._get_pipeline_ids()}) "
                     f"predictions for patient: {patient_id}, "
                     f"Aggregation type: {self.model_config.ensemble_aggregation_type.value}")
        results = [p.predict_whole_image(image_channels, voxel_spacing_mm, mask, patient_id) for p in
                   self._inference_pipelines]
        return EnsemblePipeline.aggregate_results(results, self.model_config.ensemble_aggregation_type)

    def _get_pipeline_ids(self) -> List[int]:
        return list(range(len(self._inference_pipelines)))

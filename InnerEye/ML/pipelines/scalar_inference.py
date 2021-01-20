#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from InnerEye.ML.lightning_models import ScalarLightning, load_from_checkpoint_and_adjust_for_inference
from InnerEye.ML.pipelines.inference import InferencePipelineBase
from InnerEye.ML.scalar_config import EnsembleAggregationType, ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils.model_util import get_scalar_model_inputs_and_labels


class ScalarInferencePipelineBase(InferencePipelineBase):
    """
    Base class for classification inference pipelines: inference pipelines for single and ensemble classification
    inference pipelines will inherit this.
    """

    def __init__(self, model_config: ScalarModelBase):
        super().__init__(model_config)

    def predict(self, sample: Dict[str, Any]) -> ScalarInferencePipelineBase.Result:
        raise NotImplementedError("predict must be implemented by concrete classes")

    @dataclass
    class Result:
        """
        Contains the inference results from a single pass of the inference pipeline on an input batch.
        """

        subject_ids: List[str]
        labels: torch.Tensor
        posteriors: torch.Tensor

        def __post_init__(self) -> None:
            if len(self.subject_ids) != self.labels.shape[0]:
                raise ValueError(f"Got {self.labels.shape[0]} labels for {len(self.subject_ids)} samples.")
            if len(self.subject_ids) != self.posteriors.shape[0]:
                raise ValueError(f"Got {self.posteriors.shape[0]} predictions for {len(self.subject_ids)} samples.")


class ScalarInferencePipeline(ScalarInferencePipelineBase):
    """
    Pipeline for inference from a single model on classification tasks.
    """

    def __init__(self,
                 model: ScalarLightning,
                 model_config: ScalarModelBase,
                 pipeline_id: int) -> None:
        """
        :param model: Model recovered from the checkpoint.
        :param model_config: Model configuration information.
        :param pipeline_id: ID for this pipeline (useful for ensembles).
        :return:
        """
        super().__init__(model_config)
        self.model = model
        self.pipeline_id = pipeline_id
        # Switch model to evaluation mode (if not, results will be different from what we got during training,
        # because batchnorm operates differently).
        model.eval()

    @staticmethod
    def create_from_checkpoint(path_to_checkpoint: Path,
                               config: ScalarModelBase,
                               pipeline_id: int = 0) -> Optional[ScalarInferencePipeline]:
        """
        Creates an inference pipeline from a single checkpoint.
        :param path_to_checkpoint: Path to the checkpoint to recover.
        :param config: Model configuration information.
        :param pipeline_id: ID for the pipeline to be created.
        :return: ScalarInferencePipeline if recovery from checkpoint successful. None if unsuccessful.
        """
        if not path_to_checkpoint.is_file():
            # not raising a value error here: This is used to create individual pipelines for ensembles,
            #                                   possible one model cannot be created but others can
            logging.warning(f"Could not recover model from checkpoint path {path_to_checkpoint}")
            return None
        if config.compute_mean_teacher_model:
            # TODO antonsc: Need to adjust that
            raise NotImplementedError("Mean teacher models not supported yet.")
        else:
            model = load_from_checkpoint_and_adjust_for_inference(config, path_to_checkpoint)
        assert isinstance(model, ScalarLightning)
        return ScalarInferencePipeline(model, config, pipeline_id=pipeline_id)

    def predict(self, sample: Dict[str, Any]) -> ScalarInferencePipelineBase.Result:
        """
        Runs the forward pass on a single batch.
        :param sample: Single batch of input data.
                        In the form of a dict containing at least the fields:
                        metadata, label, images, numerical_non_image_features,
                        categorical_non_image_features and segmentations.
        :return: Returns ScalarInferencePipelineBase.Result with  the subject ids, ground truth labels and predictions.
        """
        assert isinstance(self.model_config, ScalarModelBase)
        target_indices = self.model_config.get_target_indices() \
            if isinstance(self.model_config, SequenceModelBase) else []
        model_inputs_and_labels = get_scalar_model_inputs_and_labels(self.model.model,
                                                                     target_indices=target_indices,
                                                                     sample=sample)
        model_inputs_and_labels.move_to_device(self.model.device)
        with torch.no_grad():
            # This already gives the model outputs converted to posteriors
            posteriors: torch.Tensor = self.model.forward(*model_inputs_and_labels.model_inputs)
        return ScalarInferencePipelineBase.Result(subject_ids=model_inputs_and_labels.subject_ids,
                                                  labels=model_inputs_and_labels.labels,
                                                  posteriors=posteriors)


class ScalarEnsemblePipeline(ScalarInferencePipelineBase):
    """
    Pipeline for inference from an ensemble model on classification tasks. This pipeline creates models from
    multiple checkpoints and aggregates the predictions across models.
    """

    def __init__(self,
                 pipelines: List[ScalarInferencePipeline],
                 model_config: ScalarModelBase,
                 ensemble_aggregation_type: EnsembleAggregationType) -> None:
        """
        :param pipelines: A set of inference pipelines, one for each recovered checkpoint.
        :param model_config: Model configuration information.
        :param ensemble_aggregation_type: Type of aggregation to perform on the model outputs.
        :return:
        """
        super().__init__(model_config)
        self.pipelines = pipelines
        self.aggregation_type = ensemble_aggregation_type

    @staticmethod
    def create_from_checkpoint(paths_to_checkpoint: List[Path],
                               config: ScalarModelBase) -> ScalarEnsemblePipeline:
        """
        Creates an ensemble pipeline from a list of checkpoints.
        :param paths_to_checkpoint: List of paths to the checkpoints to be recovered.
        :param config: Model configuration information.
        :return:
        """
        inference_pipelines = []
        for pipeline_id, path in enumerate(paths_to_checkpoint):
            pipeline = ScalarInferencePipeline.create_from_checkpoint(path, config, pipeline_id)
            if pipeline:
                inference_pipelines.append(pipeline)
        logging.info(f"Recovered {len(inference_pipelines)} out of {len(paths_to_checkpoint)} checkpoints")

        # check if at least one inference pipeline has been created
        if not inference_pipelines:
            raise ValueError("Could not recover any of the given checkpoints")
        return ScalarEnsemblePipeline(inference_pipelines, config, config.ensemble_aggregation_type)

    def predict(self, sample: Dict[str, Any]) -> ScalarInferencePipelineBase.Result:
        """
        Performs inference on a single batch. First does the forward pass on all of the single inference pipelines,
        and then aggregates the results.
        :param sample: single batch of input data.
                        In the form of a dict containing at least the fields:
                        metadata, label, images, numerical_non_image_features,
                        categorical_non_image_features and segmentations.
        :return: Returns ScalarInferencePipelineBase.Result with the subject ids, ground truth labels and predictions.
        """
        results = [pipeline.predict(sample) for pipeline in self.pipelines]
        # Gather all model outputs into a single tensor
        posteriors = torch.stack([result.posteriors for result in results])
        posteriors_aggregated = self.aggregate_model_outputs(posteriors)
        result = ScalarInferencePipelineBase.Result(subject_ids=results[0].subject_ids,
                                                    labels=results[0].labels,
                                                    posteriors=posteriors_aggregated)
        return result

    def aggregate_model_outputs(self, model_outputs: torch.Tensor) -> torch.Tensor:
        """
        Aggregates the forward pass results from the individual models in the ensemble.
        :param model_outputs: List of model outputs for every model in the ensemble.
        (Number of ensembles) x (batch_size) x 1
        """
        # aggregate model outputs
        if self.aggregation_type == EnsembleAggregationType.Average:
            aggregated_outputs = model_outputs.mean(dim=0)
        else:
            raise NotImplementedError(f"Ensemble aggregation type {self.aggregation_type} not implemented.")

        return aggregated_outputs

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

from InnerEye.ML.pipelines.inference import InferencePipelineBase
from InnerEye.ML.reconstruction_config import ReconstructionModelBase
from InnerEye.ML.utils import model_util
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule
from InnerEye.ML.common import ModelExecutionMode

class ReconstructionInferencePipelineBase(InferencePipelineBase):
    """
    Base class for reconstruction inference pipelines: inference pipelines for single and ensemble reconstruction
    inference pipelines will inherit this.
    """

    def __init__(self, model_config: ReconstructionModelBase):
        super().__init__(model_config)

    def predict(self, sample: Dict[str, Any]) -> ReconstructionInferencePipelineBase.Result:
        raise NotImplementedError("predict must be implemented by concrete classes")

    @dataclass
    class Result:
        """
        Contains the inference results from a single pass of the inference pipeline on an input batch.
        """

        subject_ids: List[str]
        labels: torch.Tensor
        model_outputs: torch.Tensor

        def __post_init__(self) -> None:
            if len(self.subject_ids) != self.labels.shape[0]:
                raise ValueError(f"Got {self.labels.shape[0]} labels for {len(self.subject_ids)} samples.")
            if len(self.subject_ids) != self.model_outputs.shape[0]:
                print(self.model_outputs.shape)
                raise ValueError(f"Got {self.model_outputs.shape[0]} predictions for {len(self.subject_ids)} samples.")



class ReconstructionInferencePipeline(ReconstructionInferencePipelineBase):
    """
    Pipeline for inference from a single model on reconstruction tasks.
    """

    def __init__(self,
                 model: DeviceAwareModule,
                 model_config: ReconstructionModelBase,
                 epoch: int,
                 pipeline_id: int) -> None:
        """
        :param model: Model recovered from the checkpoint.
        :param model_config: Model configuration information.
        :param epoch: Epoch of the checkpoint which was recovered.
        :param pipeline_id: ID for this pipeline (useful for ensembles).
        :return:
        """
        super().__init__(model_config)
        self.model = model
        self.epoch = epoch
        self.pipeline_id = pipeline_id

        # Switch model to evaluation mode (if not, results will be different from what we got during training,
        # because batchnorm operates differently).
        model.eval()

    @staticmethod
    def create_from_checkpoint(path_to_checkpoint: Path,
                               config: ReconstructionModelBase,
                               pipeline_id: int = 0) -> Optional[ReconstructionInferencePipeline]:
        """
        Creates an inference pipeline from a single checkpoint.
        :param path_to_checkpoint: Path to the checkpoint to recover.
        :param config: Model configuration information.
        :param pipeline_id: ID for the pipeline to be created.
        :return: ReconstructionInferencePipeline if recovery from checkpoint successful. None if unsuccessful.
        """
        model_and_info = model_util.ModelAndInfo(config=config,
                                                 model_execution_mode=ModelExecutionMode.TEST,
                                                 checkpoint_path=path_to_checkpoint)

        model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
        model = model_and_info.model

        if not model_loaded:
            # not raising a value error here: This is used to create individual pipelines for ensembles,
            #                                   possible one model cannot be created but others can
            logging.warning(f"Could not recover model from checkpoint path {path_to_checkpoint}")
            return None

        # for mypy, if model has been loaded these will not be None
        assert model_and_info.checkpoint_epoch is not None

        return ReconstructionInferencePipeline(model, config, model_and_info.checkpoint_epoch, pipeline_id)

    def predict(self, sample: Dict[str, Any]) -> ReconstructionInferencePipelineBase.Result:
        """
        Runs the forward pass on a single batch.
        :param sample: Single batch of input data.
        :return: Returns ReconstructionInferencePipelineBase.Result with  the subject ids, ground truth labels and predictions.
        """
        assert isinstance(self.model_config, ReconstructionModelBase)

        labels = sample['recon']
        subjectid = sample['subjectid']
        model_output: torch.Tensor = self.model.forward(sample['kspace'])
        if isinstance(model_output, list):
            # Model output is a list if we are using data parallel. Here, this will be a degenerate list with
            # only 1 element
            model_output = torch.nn.parallel.gather(model_output, target_device=0)

        # Cast labels and model outputs back to float32, if the model had been run in mixed precision
        return ReconstructionInferencePipelineBase.Result([subjectid], labels.float(), model_output.float())


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any

import torch

from InnerEye.Common import ModelExecutionMode
from InnerEye.ML.configs.other import HelloContainer, HelloRegression
from InnerEye.ML.lightning_base import LightningModule
from InnerEye.ML.lightning_container import InnerEyeInference


class DummyEnsembleRegressionModule(HelloRegression, InnerEyeInference):
    """
    A simple 1-dim regression model intended to be trained across cross validation splits and then evaluated on test
    data using an ensemble built from the checkpoints coming from from the cross validation splits.

    To perform the ensemble inference we need to implement the methods from InnerEyeInference.
    """

    def __init__(self) -> None:
        super().__init__()

    def on_inference_start(self) -> None:
        """
        Runs initialization for everything that inference might require. This can initialize
        output files, set up metric computation, etc. This is run only once.
        """
        pass

    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        """
        Runs initialization for inference, when starting inference on a new dataset split (train/val/test).
        Depending on the settings, this can be called anywhere between 0 (no inference at all) to 3 times (inference
        on all of train/val/test split).
        :param dataset_split: Indicates whether the item comes from the training, validation or test set.
        :param is_ensemble_model: If False, the model_outputs come from an individual model. If True, the model
        outputs come from multiple models.
        """
        pass

    def inference_step(self, batch: Any, batch_idx: int, model_output: torch.Tensor) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param batch: The batch of data for which the model made a prediction.
        :param model_output: The model outputs. This would usually be a torch.Tensor, but can be any datatype.
        """
        # We don't want abstract methods here, it avoids class creation for unit tests, and we also want this
        # method to be left optional (it should be possible to also use Lightning's native test_step method)
        raise NotImplementedError("Method on_inference_start must be overwritten in a derived class.")

    def on_inference_epoch_end(self) -> None:
        """
        Called when the inference on one of the dataset splits (train/val/test) has finished.
        Depending on the settings, this can be called anywhere between 0 (no inference at all) to 3 times (inference
        on all of train/val/test split).
        """
        pass

    def on_inference_end(self) -> None:
        """
        Called when all inference epochs have finished. This can write all metrics to disk, for example. This method
        is called exactly once.
        """
        pass


class DummyEnsembleRegressionContainer(HelloContainer):
    """
    Exemplar class, based on the simple linear regression model HelloContainer, designed to show how to enable building
    an ensemble model from the checkpoints of a BYOL (Bring Your Own Lightning) cross validation model.
    """

    def __init__(self) -> None:
        super.__init__()

    def create_model(self) -> LightningModule:
        return DummyEnsembleRegressionModule()

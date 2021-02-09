#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, List

import torch
from pytorch_lightning import LightningDataModule, LightningModule

# What do we want from InnerEye?
# - datasets (optional - this means all dataset fields can potentially be left empty)
# - Do we want ensembles?
# - checkpoint recovery: checkpoints must be written to
# We are inheriting from LightningModule here, this will fail in the smaller environment
from pytorch_lightning.metrics import Metric

# Biggest problem: We don't want to rely on torch being available when submitting a job.
# A simple conda env costs 1min 30sec to create, the full one 4min 30sec in Linux.
# Could rely on only the class name when submitting to check that the model exists, skipping checks for
# the commandline overrides. Or better: Try to instantiate the class. If we can, all good. If not, just check that
# the python file exists, but proceed to submission. This will work fine for everyone working off the commandline.
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import DeepLearningConfig


# Do we want to support ensembles at inference time?

class BringYourOwnLightning(LightningModule, DeepLearningConfig):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, *args, **kwargs):
        """
        This is what is used at inference time.
        :param args:
        :param kwargs:
        :return:
        """
        # Ideally we should also move full image inference to this setup.

    def training_step(self, *args, **kwargs):
        """
        Do whatever you like here.
        :param args:
        :param kwargs:
        :return:
        """
        pass

    def get_trainer_arguments(self) -> Dict[str, Any]:
        """
        Gets additional parameters that will be passed on to the trainer.
        :return:
        """

    def get_training_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the training and validation steps.
        This must take the cross validation fold
        into account. Should those be arguments maybe? somewhat obsolete, but makes it visible.
        :return:
        """
        pass

    def get_inference_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the inference after training. By default, this returns the value
        of get_training_data_module, but you can override this to get for example full image datasets for
        segmentation models.
        This must take the cross validation fold
        into account. Should those be arguments maybe? somewhat obsolete, but makes it visible.
        :return:
        """
        # You can override this if inference uses different data, for image full images
        return self.get_training_data_module()

    def inference_metrics(self) -> List[Metric]:
        # Get metrics to compute on test set. This could be obsolete if we for example enforce that everything
        # encapsulated in inference_step
        # How do we write them to disk? They don't have a name field. Fall back to class name?
        pass

    def inference_start(self) -> None:
        """
        Runs initialization for everything that inference might require: Creating output files, for example.
        """
        pass

    def inference_step(self, item: Any, model_output: Any, mode: ModelExecutionMode) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param item: The item for which the model made a prediction.
        :param model_output: The model output. This would usually be a torch.Tensor, but can be any datatype.
        :param mode: Indicates whether the item comes from the training, validation or test set.
        :return:
        """
        # Need to make sure that both data and more are on the GPU
        pass

    def inference_end(self) -> None:
        """
        Called when the model has made predictions on all. This should write all metrics to disk.
        :return:
        """
        pass
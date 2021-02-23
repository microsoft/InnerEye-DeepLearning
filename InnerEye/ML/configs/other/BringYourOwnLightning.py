#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, Iterator, List

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


# Goals:
# run experiments by config name
# AzureML submission framework
# Consuming datasets from Azure blob storage

# Do we want to support ensembles at inference time? Not now

class BringYourOwnLightning(LightningModule, DeepLearningConfig):
    """
    Double inheritance. All files should be written to config.outputs_folder or config.logs_folder
    """
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

    def inference_start(self) -> None:
        """
        Runs initialization for everything that inference might require: Creating output files, for example.
        """
        self.metrics = whatever_list_of_metrics()

    # for dataset_split in [Train, Val, Test]
    #   model.inference_start()
    #   for item in dataloader[dataset_split]:
    #       for fold in (range(5) if is_ensemble or range(1)):
    #           model_outputs[fold] = model.forward(item)
    #       model.inference_step(item, model_outputs, dataset_split)
    #   model.inference_end()
    def inference_step(self, item: Any, model_outputs: Iterator[Any], dataset_split: ModelExecutionMode, is_ensemble_result: bool) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param item: The item for which the model made a prediction.
        :param model_output: The model output. This would usually be a torch.Tensor, but can be any datatype.
        :param mode: Indicates whether the item comes from the training, validation or test set.
        :return:
        """
        aggregate_output = None
        # Everyone has to do this. Alternative: Have a separate aggregation method
        for m in model_outputs:
            if aggregate_output is None:
                aggregate_output = m
            else:
                aggregate_output = aggregate_output + m

        # Need to make sure that both data and more are on the GPU
        model_output = self.forward(item)
        metrics = whatever(model_output, item.labels)
        # This
        self.write_output(model_output)
        pass

    def inference_end(self) -> None:
        """
        Called when the model has made predictions on all. This should write all metrics to disk.
        :return:
        """
        self.metrics.write_to_disk()
        pass

    def create_report(self) -> None:
        """
        This method should look through all files that training and inference wrote, and cook that into a
        nice human readable report. Report should go into self.outputs folder.
        """
        pass


class Container(DeepLearningConfig):

    def get_lightning_module(self) -> BringYourOwnLightning:
        pass

    def get_training_data_module(self, cross_validation_split_index: int, crossval_count: int) -> LightningDataModule:
        """
        Gets the data that is used for the training and validation steps.
        This should read a dataset from the self.local_dataset folder, but its format is up to this method here.
        This must take the cross validation fold into account.
        Should those be arguments maybe? somewhat obsolete, but makes it visible. YES.
        :return:
        """
        split = self.cross_validation_split_index
        pass

    def get_inference_data_module(self, cross_validation_split_index: int, crossval_count: int) -> LightningDataModule:
        """
        Gets the data that is used for the inference after training. By default, this returns the value
        of get_training_data_module, but you can override this to get for example full image datasets for
        segmentation models.
        This must take the cross validation fold
        into account. Should those be arguments maybe? somewhat obsolete, but makes it visible. YES
        :return:
        """
        # You can override this if inference uses different data, for image full images
        return self.get_training_data_module(cross_validation_split_index=cross_validation_split_index,
                                             crossval_count=crossval_count)

    def get_trainer_arguments(self) -> Dict[str, Any]:
        """
        Gets additional parameters that will be passed on to the trainer.
        :return:
        """


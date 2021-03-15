#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, Dict, Iterator

import torch
from pytorch_lightning import LightningDataModule, LightningModule

# Problem: We need to know
# azure_dataset_id
# model_config.get_hyperdrive_config
# model_config.perform_crossvalidation
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import DeepLearningConfig


# Biggest problem: We don't want to rely on torch being available when submitting a job.
# A simple conda env costs 1min 30sec to create, the full one 4min 30sec in Linux.
# Could rely on only the class name when submitting to check that the model exists, skipping checks for
# the commandline overrides. Or better: Try to instantiate the class. If we can, all good. If not, just check that
# the python file exists, but proceed to submission. This will work fine for everyone working off the commandline.


# flake8: noqa


# What do we want from InnerEye?
# - datasets (optional - this means all dataset fields can potentially be left empty)
# - Do we want ensembles?
# - checkpoint recovery: checkpoints must be written to
# We are inheriting from LightningModule here, this will fail in the smaller environment


# Goals:
# run experiments by config name
# AzureML submission framework
# Consuming datasets from Azure blob storage

# Do we want to support ensembles at inference time? Not now


class LightningWithInferenceMeta(type(DeepLearningConfig), type(LightningModule)):
    pass


class LightningWithInference(LightningModule, DeepLearningConfig, metaclass=LightningWithInferenceMeta):
    """
    Double inheritance. All files should be written to config.outputs_folder or config.logs_folder
    """

    def __init__(self, *args, **kwargs) -> None:
        DeepLearningConfig.__init__(self, *args, **kwargs)
        LightningModule.__init__(self)
        pass

    def forward(self, *args, **kwargs):
        """
        Run an item through the model at prediction time. This overrides LightningModule.forward, and
        has the same expected use.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("This method must be overridden in a derived class.")

    def training_step(self, *args, **kwargs):
        """
        Implements the PyTorch Lightning training step. This overrides LightningModule.training_step, and
        has the same expected use.
        """
        raise NotImplementedError("This method must be overridden in a derived class.")

    def inference_start(self) -> None:
        """
        Runs initialization for everything that the inference_step might require. This can initialize
        output files, set up metric computation, etc.
        """
        pass

    # for dataset_split in [Train, Val, Test]
    #   model.inference_start()
    #   for item in dataloader[dataset_split]:
    #       for fold in (range(5) if is_ensemble or range(1)):
    #           model_outputs[fold] = model.forward(item)
    #       model.inference_step(item, model_outputs, dataset_split, is_ensemble_result=is_ensemble_result)
    #   model.inference_end()
    def inference_step(self, item: Any, model_outputs: Iterator[Any], dataset_split: ModelExecutionMode,
                       is_ensemble_result: bool) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param item: The item for which the model made a prediction.
        :param model_outputs: The model output. This would usually be a torch.Tensor, but can be any datatype.
        :param dataset_split: Indicates whether the item comes from the training, validation or test set.
        :param is_ensemble_result: If False, the model_outputs come from an individual model If True, the model
        outputs come from multiple models.
        """
        pass

    def aggregate_model_outputs(self, model_outputs: Iterator[torch.Tensor]) -> torch.Tensor:
        """
        Aggregates the outputs of multiple models when using an ensemble model. In the default implementation,
        this averages the tensors coming from all the models.
        :param model_outputs: An iterator over the model outputs for all ensemble members.
        :return: The aggregate model outputs.
        """
        aggregate_output = None
        count = 0
        for m in model_outputs:
            count += 1
            if aggregate_output is None:
                aggregate_output = m
            else:
                aggregate_output = aggregate_output + m
        aggregate_output = aggregate_output / count
        return aggregate_output

    def inference_end(self) -> None:
        """
        Called when the model has made predictions on all. This can write all metrics to disk, for example.
        """
        pass

    def create_report(self) -> None:
        """
        This method should look through all files that training and inference wrote, and cook that into a
        nice human readable report. Report should go into self.outputs folder.
        """
        pass


class LightningContainer:

    def __init__(self):
        super().__init__()
        self._lightning_module = None

    @property
    def lightning_module(self) -> LightningWithInference:
        """
        Returns that PyTorch Lightning module that the present container object manages.
        :return: A PyTorch Lightning module
        """
        if self._lightning_module is None:
            raise ValueError("No Lightning module has been set yet.")
        return self._lightning_module

    @lightning_module.setter
    def lightning_module(self, value: LightningWithInference) -> None:
        self._lightning_module = value

    def create_lightning_module(self) -> LightningWithInference:
        pass

    def get_training_data_module(self, crossval_index: int, crossval_count: int) -> LightningDataModule:
        """
        Gets the data that is used for the training and validation steps.
        This should read a dataset from the self.local_dataset folder or download from a web location.
        The format of the data is not specified any further.
        The method must take cross validation into account, and ensure that logic to create training and validation
        sets takes cross validation with a given number of splits is correctly taken care of.
        :return: A LightningDataModule
        """
        pass

    def get_inference_data_module(self, crossval_index: int, crossval_count: int) -> LightningDataModule:
        """
        Gets the data that is used for the inference after training. By default, this returns the value
        of get_training_data_module, but you can override this to get for example full image datasets for
        segmentation models.
        This should read a dataset from the self.local_dataset folder or download from a web location.
        The format of the data is not specified any further.
        The method must take cross validation into account, and ensure that logic to create training and validation
        sets takes cross validation with a given number of splits is correctly taken care of.
        :return: A LightningDataModule
        """
        # You can override this if inference uses different data, for example segmentation models use
        # full images rather than equal sized crops.
        return self.get_training_data_module(crossval_index=crossval_index, crossval_count=crossval_count)

    def get_trainer_arguments(self) -> Dict[str, Any]:
        """
        Gets additional parameters that will be passed on to the PL trainer.
        """
        return dict()

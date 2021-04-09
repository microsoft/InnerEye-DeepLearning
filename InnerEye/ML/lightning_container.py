#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import abc
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import param
import torch
from pytorch_lightning import LightningDataModule, LightningModule
# Problem: We need to know
# azure_dataset_id
# model_config.get_hyperdrive_config
# model_config.perform_crossvalidation
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from InnerEye.Common.generic_parsing import GenericConfig, create_from_matching_params
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import DatasetParams, EssentialParams, OptimizerParams, OutputParams, \
    TrainerParams
# Do we want to support ensembles at inference time? Not now
from InnerEye.ML.utils import model_util
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp


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

# rename from inference_... to test_step
# How to define parameters?
# Can we simplify score.py by re-using code here?
# automatic regression testing

# We can restrict inference to only the test set for pretty much all models. Only in segmentation models we
# are presently doing something different than in training.
# If that is not enough: Can pass in inference mode (single model / ensemble) and which dataset we are running on.

class LightningInference(abc.ABC):
    """
    A base class that defines the methods that need to be present for doing inference on a trained model.
    The inference code calls the methods in this order:

    model.inference_start()
    for dataset_split in [Train, Val, Test]
        model.on_inference_epoch_start(dataset_split, is_ensemble_model=False)
        for batch_idx, item in enumerate(dataloader[dataset_split])):
            model_outputs = model.forward(item)
            model.inference_step(item, batch_idx, model_outputs)
        model.on_inference_epoch_end()
    model.on_inference_end()
    """

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

    def aggregate_ensemble_model_outputs(self, model_outputs: Iterator[torch.Tensor]) -> torch.Tensor:
        """
        Aggregates the outputs of multiple models when using an ensemble model. In the default implementation,
        this averages the tensors coming from all the models.
        :param model_outputs: An iterator over the model outputs for all ensemble members.
        :return: The aggregate model outputs.
        """
        aggregate_output: Optional[torch.Tensor] = None
        count = 0
        for m in model_outputs:
            count += 1
            if aggregate_output is None:
                aggregate_output = m
            else:
                aggregate_output += m
        if count == 0 or aggregate_output is None:
            raise ValueError("There were no results to aggregate.")
        aggregate_output /= count
        return aggregate_output


class LightningWithInference(LightningModule, LightningInference):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        LightningModule.__init__(self, *args, **kwargs)
        # These 3 fields get populated from the enclosing LightningContainer, in create_lightning_module_and_store
        self.optimizer_params = OptimizerParams()
        self.output_params = OutputParams()
        self.trainer_params = TrainerParams()
        pass

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Run an item through the model at prediction time. This overrides LightningModule.forward, and
        has the same expected use.
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError("This method must be overridden in a derived class.")

    def training_step(self, *args: Any, **kwargs: Any) -> Any:
        """
        Implements the PyTorch Lightning training step. This overrides LightningModule.training_step, and
        has the same expected use.
        """
        raise NotImplementedError("This method must be overridden in a derived class.")

    def create_report(self) -> None:
        """
        This method should look through all files that training and inference wrote, and cook that into a
        nice human readable report. Report should go into self.outputs folder.
        """
        pass

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This is the default implementation of the method that provides the optimizer and LR scheduler for
        PyTorch Lightning. It reads out the optimizer and scheduler settings from the model fields,
        and creates the two objects.
        Override this method for full flexibility to define any optimizer and scheduler.
        :return: A tuple of (optimizer, LR scheduler)
        """
        optimizer = model_util.create_optimizer(self.optimizer_params, self.parameters())
        l_rate_scheduler = SchedulerWithWarmUp(self.optimizer_params, optimizer,
                                               num_epochs=self.trainer_params.num_epochs)
        return [optimizer], [l_rate_scheduler]

    def close_all_loggers(self) -> None:
        """
        This method should close all objects that were used during training to write additional data.
        """
        pass

    @property
    def outputs_folder(self) -> Path:
        """
        Gets the folder into which all output of the model training should be written. Output files in this folder
        will be available in AzureML at the end of training.
        """
        return self.output_params.outputs_folder

    @property
    def logs_folder(self) -> Path:
        """
        Gets the folder into which all log files from model training should be written. Log files will be streamed
        to AzureML during training already.
        """
        return self.output_params.logs_folder


class LightningContainer(GenericConfig,
                         EssentialParams,
                         DatasetParams,
                         OutputParams,
                         TrainerParams,
                         OptimizerParams):

    def __init__(self) -> None:
        super().__init__()
        self._model: Optional[LightningWithInference] = None
        self._model_name = type(self).__name__

    def setup(self) -> None:
        """
        This method is called as one of the first operations of the training/testing workflow, before any other
        operations on the present object. At the point when called, the dataset is already available in
        the location given by self.local_dataset. Use this method to prepare datasets or data loaders, for example.
        """
        pass

    def create_model(self) -> LightningWithInference:
        """
        This method must create the actual Lightning model that will be trained. It can read out parameters from the
        container and pass them into the model, for example.
        """
        pass

    def get_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the training, validation, and test steps.
        This should read a dataset from the self.local_dataset folder or download from a web location.
        The format of the data is not specified any further.
        The method must take cross validation into account, and ensure that logic to create training and validation
        sets takes cross validation with a given number of splits is correctly taken care of.
        :return: A LightningDataModule
        """
        pass

    def get_inference_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used to evaluate the trained model. By default, this returns the value
        of get_data_module(), but you can override this to get for example full image datasets for
        segmentation models.
        This should read a dataset from the self.local_dataset folder or download from a web location.
        The format of the data is not specified any further.
        The method must take cross validation into account, and ensure that logic to create training and validation
        sets takes cross validation with a given number of splits is correctly taken care of.
        :return: A LightningDataModule
        """
        # You can override this if inference uses different data, for example segmentation models use
        # full images rather than equal sized crops.
        return self.get_data_module()

    def get_trainer_arguments(self) -> Dict[str, Any]:
        """
        Gets additional parameters that will be passed on to the PL trainer.
        """
        return dict()

    def before_training_on_rank_zero(self) -> None:
        """
        A hook that will be called before starting model training, before creating the Lightning Trainer object.
        In distributed training, this is only run on rank zero. It is executed after the before_training_on_all_ranks
        hook.
        """
        pass

    def before_training_on_all_ranks(self) -> None:
        """
        A hook that will be called before starting model training.
        In distributed training, this hook will be called on all ranks. It is executed before the
        the before_training_on_rank_zero hook.
        """
        pass

    # The code from here onwards does not need to be modified.

    @property
    def model(self) -> LightningWithInference:
        """
        Returns the PyTorch Lightning module that the present container object manages.
        :return: A PyTorch Lightning module
        """
        if self._model is None:
            raise ValueError("No Lightning module has been set yet.")
        return self._model

    def create_lightning_module_and_store(self) -> None:
        """
        Creates the Lightning model by calling `create_lightning_module` and stores it in the `lightning_module`
        property.
        """
        self._model = self.create_model()
        self._model.output_params = create_from_matching_params(self, OutputParams)
        self._model.optimizer_params = create_from_matching_params(self, OptimizerParams)
        self._model.trainer_params = create_from_matching_params(self, TrainerParams)

    def __str__(self) -> str:
        """Returns a string describing the present object, as a list of key: value strings."""
        arguments_str = "\nContainer:\n"
        # Avoid callable params, the bindings that are printed out can be humongous.
        # Avoid dataframes
        skip_params = {name for name, value in self.param.params().items()
                       if isinstance(value, (param.Callable, param.DataFrame))}
        for key, value in self.param.get_param_values():
            if key not in skip_params:
                arguments_str += f"\t{key:40}: {value}\n"
        # Print out all other separate vars that are not under the guidance of the params library,
        # skipping the two that are introduced by params
        skip_vars = {"param", "initialized"}
        for key, value in vars(self).items():
            if key not in skip_vars and key[0] != "_":
                arguments_str += f"\t{key:40}: {value}\n"
        return arguments_str

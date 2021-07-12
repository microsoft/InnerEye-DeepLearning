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
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from azureml.core import ScriptRunConfig
from azureml.train.hyperdrive import GridParameterSampling, HyperDriveConfig, PrimaryMetricGoal, choice

from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.generic_parsing import GenericConfig, create_from_matching_params
from InnerEye.Common.metrics_constants import TrackedMetrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.deep_learning_config import DatasetParams, OptimizerParams, OutputParams, TrainerParams, \
    WorkflowParams, load_checkpoint
from InnerEye.ML.utils import model_util
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.run_recovery import RunRecovery


class InnerEyeInference(abc.ABC):
    """
    A base class that defines the methods that need to be present for doing inference on a trained model. This
    form of inference is slightly different from what PyTorch Lightning does in its `Trainer.test` method. In
    particular, this inference can be executed on any of the training, validation, or test set.

    The inference code calls the methods in this order:

    model.on_inference_start()
    for dataset_split in [Train, Val, Test]
        model.on_inference_start_dataset(dataset_split, is_ensemble_model=False)
        for batch_idx, batch in enumerate(dataloader[dataset_split])):
            posteriors = model.forward(item)
            model.record_posteriors(batch, batch_idx, posteriors)
        model.on_inference_end_dataset()
    model.on_inference_end()
    """

    def on_inference_start(self) -> None:
        """
        Runs initialization for everything that inference might require. This can initialize
        output files, set up metric computation, etc. This is run only once.
        """
        pass

    def on_inference_start_dataset(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        """
        Runs initialization for inference, when starting inference on a new dataset split (train/val/test).
        Depending on the settings, this can be called anywhere between 0 (no inference at all) to 3 times (inference
        on all of train/val/test split).
        :param dataset_split: Indicates whether the item comes from the training, validation or test set.
        :param is_ensemble_model: If False, the model_outputs come from an individual model. If True, the model
        outputs come from multiple models.
        """
        pass

    def record_posteriors(self, batch: Any, batch_idx: int, posteriors: torch.Tensor) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param batch: The batch of data for which the model made a prediction.
        :param posteriors: The posteriors output by the model.
        """
        # We don't want abstract methods here, it avoids class creation for unit tests, and we also want this
        # method to be left optional (it should be possible to also use Lightning's native test_step method)
        raise NotImplementedError("Method on_inference_start must be overwritten in a derived class.")

    def on_inference_end_dataset(self) -> None:
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

    @staticmethod
    def aggregate_ensemble_model_outputs(model_outputs: Iterator[torch.Tensor]) -> torch.Tensor:
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


class LightningModuleWithOptimizer(LightningModule):
    """
    A base class that supplies a method to configure optimizers and LR schedulers. To use this in your model,
    inherit from this class instead of from LightningModule.
    If this class is used, all configuration options for the optimizers and LR schedulers will be also available as
    commandline arguments (for example, you can supply the InnerEye runner with "--l_rate=1e-2" to change the learning
    rate.
    """
    # These fields will be set by the LightningContainer when the model is created.
    _optimizer_params = OptimizerParams()
    _trainer_params = TrainerParams()

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This is the default implementation of the method that provides the optimizer and LR scheduler for
        PyTorch Lightning. It reads out the optimizer and scheduler settings from the model fields,
        and creates the two objects.
        Override this method for full flexibility to define any optimizer and scheduler.
        :return: A tuple of (optimizer, LR scheduler)
        """
        optimizer = model_util.create_optimizer(self._optimizer_params, self.parameters())
        l_rate_scheduler = SchedulerWithWarmUp(self._optimizer_params, optimizer,
                                               num_epochs=self._trainer_params.num_epochs)
        return [optimizer], [l_rate_scheduler]


class LightningContainer(GenericConfig,
                         WorkflowParams,
                         DatasetParams,
                         OutputParams,
                         TrainerParams,
                         OptimizerParams):
    """
    A LightningContainer contains all information to train a user-specified PyTorch Lightning model. The model that
    should be trained is returned by the `create_model` method. The training data must be returned in the form of
    a LightningDataModule, by the `get_data_module` method.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model: Optional[LightningModule] = None
        self._model_name = type(self).__name__
        self.extra_downloaded_run_id: Optional[RunRecovery] = None
        self.num_nodes = 1

    def validate(self) -> None:
        WorkflowParams.validate(self)
        OptimizerParams.validate(self)

    def setup(self) -> None:
        """
        This method is called as one of the first operations of the training/testing workflow, before any other
        operations on the present object. At the point when called, the dataset is already available in
        the location given by self.local_dataset. Use this method to prepare datasets or data loaders, for example.
        """
        pass

    def create_model(self) -> LightningModule:
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
        Because the method deals with data loaders, not loaded data, we cannot check automatically that cross validation
        is handled correctly within the base class, i.e. if the cross validation split is not handled in the method then
        nothing will fail, but each child run will be identical since they will each be given the full dataset.
        :return: A LightningDataModule
        """
        return None  # type: ignore

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
        Gets additional parameters that will be passed on to the PyTorch Lightning trainer.
        """
        return dict()

    def get_parameter_search_hyperdrive_config(self, _: ScriptRunConfig) -> HyperDriveConfig:  # type: ignore
        """
        Parameter search is not implemented. It should be implemented in a sub class if needed.
        """
        raise NotImplementedError("Parameter search is not implemented. It should be implemented in a sub class if needed.")

    def create_report(self) -> None:
        """
        This method is called after training and testing has been completed. It can aggregate all files that were
        written during training and testing, and compile them into some helpful overarching output.
        The report should be written to self.
        """
        pass

    def before_training_on_global_rank_zero(self) -> None:
        """
        A hook that will be called before starting model training, before creating the Lightning Trainer object.
        In distributed training, this is only run on global rank zero (i.e, on the process that runs on node 0, GPU 0).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def before_training_on_local_rank_zero(self) -> None:
        """
        A hook that will be called before starting model training.
        In distributed training, this hook will be called once per node (i.e., whenever the LOCAL_RANK environment
        variable is zero).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def before_training_on_all_ranks(self) -> None:
        """
        A hook that will be called before starting model training.
        In distributed training, this hook will be called on all ranks (i.e., once per GPU).
        The order in which hooks are called is: before_training_on_global_rank_zero, before_training_on_local_rank_zero,
        before_training_on_all_ranks.
        """
        pass

    def load_checkpoint_and_modify(self, path_to_checkpoint: Path) -> Dict[str, Any]:
        """
        This method is called when a file with weights for network initialization is supplied at container level,
        in the self.weights_url or self.local_weights_path fields. It can load that file as a Torch checkpoint,
        and rename parameters.

        By default, uses torch.load to read and return the state dict from the checkpoint file, and does no modification
        of the checkpoint file.

        Overloading this function:
        When weights_url or local_weights_path is set, the file downloaded may not be in the exact
        format expected by the model's load_state_dict() - for example, pretrained Imagenet weights for networks
        may have mismatched layer names in different implementations.
        In such cases, you can overload this function to extract the state dict from the checkpoint.

        NOTE: The model checkpoint will be loaded using the torch function load_state_dict() with argument strict=False,
        so extra care needs to be taken to check that the state dict is valid.
        Check the logs for warnings related to missing and unexpected keys.
        See https://pytorch.org/tutorials/beginner/saving_loading_models.html#warmstarting-model-using-parameters
        -from-a-different-model
        for an explanation on why strict=False is useful when loading parameters from other models.
        :param path_to_checkpoint: Path to the checkpoint file.
        :return: Dictionary with model and optimizer state dicts. The dict should have at least the following keys:
        1. Key ModelAndInfo.MODEL_STATE_DICT_KEY and value set to the model state dict.
        2. Key ModelAndInfo.EPOCH_KEY and value set to the checkpoint epoch.
        Other (optional) entries corresponding to keys ModelAndInfo.OPTIMIZER_STATE_DICT_KEY and
        ModelAndInfo.MEAN_TEACHER_STATE_DICT_KEY are also supported.
        """
        return load_checkpoint(path_to_checkpoint=path_to_checkpoint, use_gpu=self.use_gpu)

    # The code from here on does not need to be modified.

    @property
    def model(self) -> LightningModule:
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
        if isinstance(self._model, LightningModuleWithOptimizer):
            self._model._optimizer_params = create_from_matching_params(self, OptimizerParams)
            self._model._trainer_params = create_from_matching_params(self, TrainerParams)

    def get_cross_validation_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Returns a configuration for AzureML Hyperdrive that varies the cross validation split index.
        Because this adds a val/Loss metric it is important that when subclassing LightningContainer
        your implementeation of LightningModule logs val/Loss. There is an example of this in 
        HelloRegression's validation_step method.
        :param run_config: The AzureML run configuration object that training for an individual model.
        :return: A hyperdrive configuration object.
        """
        return HyperDriveConfig(
            run_config=run_config,
            hyperparameter_sampling=GridParameterSampling(
                parameter_space={
                    CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: choice(list(range(self.number_of_cross_validation_splits)))
                }),
            primary_metric_name=TrackedMetrics.Val_Loss.value,
            primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
            max_total_runs=self.number_of_cross_validation_splits
        )

    def get_hyperdrive_config(self, run_config: ScriptRunConfig) -> HyperDriveConfig:
        """
        Returns the HyperDrive config for either parameter search or cross validation
        (if number_of_cross_validation_splits > 1).
        :param run_config: AzureML estimator
        :return: HyperDriveConfigs
        """
        if self.perform_cross_validation:
            return self.get_cross_validation_hyperdrive_config(run_config)
        else:
            return self.get_parameter_search_hyperdrive_config(run_config)

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

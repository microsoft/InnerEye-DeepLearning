#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import param
from enum import Enum, unique
from pandas import DataFrame
from param import Parameterized
from pathlib import Path
from typing import Any, Dict, List, Optional

from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, RUN_CONTEXT, is_offline_run_context
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import ModelProcessing, is_windows
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR, DEFAULT_LOGS_DIR_NAME
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.type_annotations import PathOrString, TupleFloat2
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode, create_unique_timestamp_id, \
    get_best_checkpoint_path, get_recovery_checkpoint_path

# A folder inside of the outputs folder that will contain all information for running the model in inference mode

FINAL_MODEL_FOLDER = "final_model"
FINAL_ENSEMBLE_MODEL_FOLDER = "final_ensemble_model"

# The checkpoints must be stored inside of the final model folder, if we want to avoid copying
# them before registration.
CHECKPOINT_FOLDER = "checkpoints"
VISUALIZATION_FOLDER = "visualizations"
EXTRA_RUN_SUBFOLDER = "extra_run_id"

ARGS_TXT = "args.txt"
WEIGHTS_FILE = "weights.pth"


@unique
class LRWarmUpType(Enum):
    """
    Supported LR warm up types for model training
    """
    NoWarmUp = "NoWarmUp"
    Linear = "Linear"


@unique
class LRSchedulerType(Enum):
    """
    Supported lr scheduler types for model training
    """
    Exponential = "Exponential"
    Step = "Step"
    Polynomial = "Polynomial"
    Cosine = "Cosine"
    MultiStep = "MultiStep"


@unique
class OptimizerType(Enum):
    """
    Supported optimizers for model training
    """
    Adam = "Adam"
    AMSGrad = "AMSGrad"
    SGD = "SGD"
    RMSprop = "RMSprop"


@unique
class ModelCategory(Enum):
    """
    Describes the different high-level model categories that the codebase supports.
    """
    Segmentation = "Segmentation"  # All models that perform segmentation: Classify each voxel in the input image.
    Classification = "Classification"  # All models that perform classification
    Regression = "Regression"  # All models that perform regression

    @property
    def is_scalar(self) -> bool:
        """
        Return True if the current ModelCategory is either Classification or Regression
        """
        return self in [ModelCategory.Classification, ModelCategory.Regression]


@unique
class MultiprocessingStartMethod(Enum):
    """
    Different methods for starting data loader processes.
    """
    fork = "fork"
    forkserver = "forkserver"
    spawn = "spawn"


class TemperatureScalingConfig(Parameterized):
    """High level config to encapsulate temperature scaling parameters"""
    lr: float = param.Number(default=0.002, bounds=(0, None),
                             doc="The learning rate to use for the optimizer used to learn the "
                                 "temperature scaling parameter")
    max_iter: int = param.Number(default=50, bounds=(1, None),
                                 doc="The maximum number of optimization iterations to use in order to "
                                     "learn the temperature scaling parameter")
    ece_num_bins: int = param.Number(default=15, bounds=(1, None),
                                     doc="Number of bins to use when computing the "
                                         "Expected Calibration Error")


class DeepLearningFileSystemConfig(Parameterized):
    """High level config to abstract the file system related configs for deep learning models"""
    outputs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                               doc="The folder where all training and test outputs should go.")
    logs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                            doc="The folder for all log files and Tensorboard event files")
    project_root: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                             doc="The root folder for the codebase that triggers the training run.")
    run_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                           doc="The folder that contains outputs and the logs subfolder.")

    @staticmethod
    def create(project_root: Path,
               is_offline_run: bool,
               model_name: str,
               output_to: Optional[str] = None) -> DeepLearningFileSystemConfig:
        """
        Creates a new object that holds output folder configurations. When running inside of AzureML, the output
        folders will be directly under the project root. If not running inside AzureML, a folder with a timestamp
        will be created for all outputs and logs.
        :param project_root: The root folder that contains the code that submitted the present training run.
        When running inside the InnerEye repository, it is the git repo root. When consuming InnerEye as a package,
        this should be the root of the source code that calls the package.
        :param is_offline_run: If true, this is a run outside AzureML. If False, it is inside AzureML.
        :param model_name: The name of the model that is trained. This is used to generate a run-specific output
        folder.
        :param output_to: If provided, the output folders will be created as a subfolder of this argument. If not
        given, the output folders will be created inside of the project root.
        """
        if not project_root.is_absolute():
            raise ValueError(f"The project root is required to be an absolute path, but got {project_root}")

        if is_offline_run or output_to:
            if output_to:
                logging.info(f"All results will be written to the specified output folder {output_to}")
                root = Path(output_to).absolute()
            else:
                logging.info("All results will be written to a subfolder of the project root folder.")
                root = project_root.absolute() / DEFAULT_AML_UPLOAD_DIR
            timestamp = create_unique_timestamp_id()
            run_folder = root / f"{timestamp}_{model_name}"
            outputs_folder = run_folder
            logs_folder = run_folder / DEFAULT_LOGS_DIR_NAME
        else:
            logging.info("Running inside AzureML.")
            logging.info("All results will be written to a subfolder of the project root folder.")
            run_folder = project_root
            outputs_folder = project_root / DEFAULT_AML_UPLOAD_DIR
            logs_folder = project_root / DEFAULT_LOGS_DIR_NAME
        logging.info(f"Run outputs folder: {outputs_folder}")
        logging.info(f"Logs folder: {logs_folder}")
        return DeepLearningFileSystemConfig(
            outputs_folder=outputs_folder,
            logs_folder=logs_folder,
            project_root=project_root,
            run_folder=run_folder
        )

    def add_subfolder(self, subfolder: str) -> DeepLearningFileSystemConfig:
        """
        Creates a new output folder configuration, where both outputs and logs go into the given subfolder inside
        the present outputs folder.
        :param subfolder: The subfolder that should be created.
        :return:
        """
        if self.run_folder:
            outputs_folder = self.run_folder / subfolder
            logs_folder = self.run_folder / subfolder / DEFAULT_LOGS_DIR_NAME
            outputs_folder.mkdir(parents=True, exist_ok=True)
            logs_folder.mkdir(parents=True, exist_ok=True)
            return DeepLearningFileSystemConfig(
                outputs_folder=outputs_folder,
                logs_folder=logs_folder,
                project_root=self.project_root
            )
        raise ValueError("This method should only be called for runs outside AzureML, when the logs folder is "
                         "inside the outputs folder.")


class WorkflowParams(param.Parameterized):
    """
    This class contains all parameters that affect how the whole training and testing workflow is executed.
    """
    random_seed: int = param.Integer(42, doc="The seed to use for all random number generators.")
    number_of_cross_validation_splits: int = param.Integer(0, bounds=(0, None),
                                                           doc="Number of cross validation splits for k-fold cross "
                                                               "validation")
    cross_validation_split_index: int = param.Integer(DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, bounds=(-1, None),
                                                      doc="The index of the cross validation fold this model is "
                                                          "associated with when performing k-fold cross validation")
    inference_on_train_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on training set after training.")
    inference_on_val_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on validation set after training.")
    inference_on_test_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on test set after training.")
    ensemble_inference_on_train_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on the training set after ensemble training.")
    ensemble_inference_on_val_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on validation set after ensemble training.")
    ensemble_inference_on_test_set: bool = \
        param.Boolean(None,
                      doc="If set, enable/disable full image inference on test set after ensemble training.")
    weights_url: str = param.String(doc="If provided, a url from which weights will be downloaded and used for model "
                                        "initialization.")
    local_weights_path: Optional[Path] = param.ClassSelector(class_=Path,
                                                             default=None,
                                                             allow_None=True,
                                                             doc="The path to the weights to use for model "
                                                                 "initialization, when training outside AzureML.")
    generate_report: bool = param.Boolean(default=True,
                                          doc="If True (default), write a modelling report in HTML format. If False,"
                                              "do not write that report.")
    # The default multiprocessing start_method in both PyTorch and the Python standard library is "fork" for Linux and
    # "spawn" (the only available method) for Windows. There is some evidence that using "forkserver" on Linux
    # can reduce the chance of stuck jobs.
    multiprocessing_start_method: MultiprocessingStartMethod = \
        param.ClassSelector(class_=MultiprocessingStartMethod,
                            default=(MultiprocessingStartMethod.spawn if is_windows()
                                     else MultiprocessingStartMethod.fork),
                            doc="Method to be used to start child processes in pytorch. Should be one of forkserver, "
                                "fork or spawn. If not specified, fork is used on Linux and spawn on Windows. "
                                "Set to forkserver as a possible remedy for stuck jobs.")
    monitoring_interval_seconds: int = param.Integer(0, doc="Seconds delay between logging GPU/CPU resource "
                                                            "statistics. If 0 or less, do not log any resource "
                                                            "statistics.")
    regression_test_folder: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="A path to a folder that contains a set of files. At the end of training and "
                                "model evaluation, all files given in that folder must be present in the job's output "
                                "folder, and their contents must match exactly. When running in AzureML, you need to "
                                "ensure that this folder is part of the snapshot that gets uploaded. The path should "
                                "be relative to the repository root directory.")

    def validate(self) -> None:
        if self.weights_url and self.local_weights_path:
            raise ValueError("Cannot specify both local_weights_path and weights_url.")

        if self.number_of_cross_validation_splits == 1:
            raise ValueError("At least two splits required to perform cross validation, but got "
                             f"{self.number_of_cross_validation_splits}. To train without cross validation, set "
                             "number_of_cross_validation_splits=0.")
        if 0 < self.number_of_cross_validation_splits <= self.cross_validation_split_index:
            raise ValueError(f"Cross validation split index is out of bounds: {self.cross_validation_split_index}, "
                             f"which is invalid for CV with {self.number_of_cross_validation_splits} splits.")
        elif self.number_of_cross_validation_splits == 0 and self.cross_validation_split_index != -1:
            raise ValueError(f"Cross validation split index must be -1 for a non cross validation run, "
                             f"found number_of_cross_validation_splits = {self.number_of_cross_validation_splits} "
                             f"and cross_validation_split_index={self.cross_validation_split_index}")

    """ Defaults for when to run inference in the absence of any command line switches. """
    INFERENCE_DEFAULTS: Dict[ModelProcessing, Dict[ModelExecutionMode, bool]] = {
        ModelProcessing.DEFAULT: {
            ModelExecutionMode.TRAIN: False,
            ModelExecutionMode.TEST: True,
            ModelExecutionMode.VAL: True,
        },
        ModelProcessing.ENSEMBLE_CREATION: {
            ModelExecutionMode.TRAIN: False,
            ModelExecutionMode.TEST: True,
            ModelExecutionMode.VAL: False,
        }
    }

    """ Mapping from ModelProcesing and ModelExecutionMode to command line switch. """
    INFERENCE_OPTIONS: Dict[ModelProcessing, Dict[ModelExecutionMode, str]] = {
        ModelProcessing.DEFAULT: {
            ModelExecutionMode.TRAIN: 'inference_on_train_set',
            ModelExecutionMode.TEST: 'inference_on_test_set',
            ModelExecutionMode.VAL: 'inference_on_val_set',
        },
        ModelProcessing.ENSEMBLE_CREATION: {
            ModelExecutionMode.TRAIN: 'ensemble_inference_on_train_set',
            ModelExecutionMode.TEST: 'ensemble_inference_on_test_set',
            ModelExecutionMode.VAL: 'ensemble_inference_on_val_set',
        }
    }

    def run_perform_inference(self, model_proc: ModelProcessing, data_split: ModelExecutionMode) -> bool:
        """
        Returns True if inference is required for this model_proc and data_split.

        :param model_proc: Whether we are testing an ensemble or single model.
        :param data_split: Indicates which of the 3 sets (training, test, or validation) is being processed.
        :return: True if inference required.
        """
        inference_option = WorkflowParams.INFERENCE_OPTIONS[model_proc][data_split]
        inference_option_val = getattr(self, inference_option)

        if inference_option_val is not None:
            return inference_option_val

        return WorkflowParams.INFERENCE_DEFAULTS[model_proc][data_split]

    @property
    def is_offline_run(self) -> bool:
        """
        Returns True if the run is executing outside AzureML, or False if inside AzureML.
        """
        return is_offline_run_context(RUN_CONTEXT)

    @property
    def perform_cross_validation(self) -> bool:
        """
        True if cross validation will be be performed as part of the training procedure.
        :return:
        """
        return self.number_of_cross_validation_splits > 1

    def get_effective_random_seed(self) -> int:
        """
        Returns the random seed set as part of this configuration. If the configuration corresponds
        to a cross validation split, then the cross validation fold index will be added to the
        set random seed in order to return the effective random seed.
        :return:
        """
        seed = self.random_seed
        if self.perform_cross_validation:
            # offset the random seed based on the cross validation split index so each
            # fold has a different initial random state.
            seed += self.cross_validation_split_index
        return seed


class DatasetParams(param.Parameterized):
    azure_dataset_id: str = param.String(doc="If provided, the ID of the dataset to use when running in AzureML. "
                                             "This dataset must exist as a folder of the same name in the 'datasets' "
                                             "container in the datasets storage account. This dataset will be mounted "
                                             "and made available at the 'local_dataset' path when running in AzureML.")
    local_dataset: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None, allow_None=True,
                            doc="The path of the dataset to use, when training is running outside Azure.")
    extra_azure_dataset_ids: List[str] = \
        param.List(default=[], allow_None=False,
                   doc="This can be used to feed in additional datasets to your custom datamodules. These will be"
                       "mounted and made available as a list of paths in 'extra_local_datasets' when running in AML.")
    extra_local_dataset_paths: List[Path] = param.List(class_=Path, default=[], allow_None=False,
                                                       doc="This can be used to feed in additional datasets "
                                                           "to your custom datamodules when running outside of Azure "
                                                           "AML.")
    dataset_mountpoint: str = param.String(doc="The path at which the AzureML dataset should be made available via "
                                               "mounting or downloading. This only affects jobs running in AzureML."
                                               "If empty, use a random mount/download point.")
    extra_dataset_mountpoints: List[str] = \
        param.List(default=[], allow_None=False,
                   doc="The mounting points for the datasets given in extra_azure_dataset_ids, when running in "
                       "AzureML. Use an empty string for all datasets where a randomly chosen mount/download point "
                       "should be used.")

    def validate(self) -> None:
        if not self.azure_dataset_id and self.local_dataset is None:
            raise ValueError("Either of local_dataset or azure_dataset_id must be set.")

        if self.all_dataset_mountpoints() and len(self.all_azure_dataset_ids()) != len(self.all_dataset_mountpoints()):
            raise ValueError(f"Expected the number of azure datasets to equal the number of mountpoints, "
                             f"got datasets [{','.join(self.all_azure_dataset_ids())}] "
                             f"and mountpoints [{','.join(self.all_dataset_mountpoints())}]")

    def all_azure_dataset_ids(self) -> List[str]:
        """
        Returns a list with all azure dataset IDs that are specified in self.azure_dataset_id and
        self.extra_azure_dataset_ids
        """
        if not self.azure_dataset_id:
            return self.extra_azure_dataset_ids
        else:
            return [self.azure_dataset_id] + self.extra_azure_dataset_ids

    def all_dataset_mountpoints(self) -> List[str]:
        """
        Returns a list with all dataset mount points that are specified in self.dataset_mountpoint and
        self.extra_dataset_mountpoints
        """
        if not self.dataset_mountpoint:
            return self.extra_dataset_mountpoints
        else:
            return [self.dataset_mountpoint] + self.extra_dataset_mountpoints


class OutputParams(param.Parameterized):
    output_to: str = param.String(default="",
                                  doc="If provided, the run outputs will be written to the given folder. If not "
                                      "provided, outputs will go into a subfolder of the project root folder.")
    file_system_config: DeepLearningFileSystemConfig = param.ClassSelector(default=DeepLearningFileSystemConfig(),
                                                                           class_=DeepLearningFileSystemConfig,
                                                                           instantiate=False,
                                                                           doc="File system related configs")
    _model_name: str = param.String("", doc="The human readable name of the model (for example, Liver). This is "
                                            "usually set from the class name.")

    @property
    def model_name(self) -> str:
        """
        Gets the human readable name of the model (e.g., Liver). This is usually set from the class name.
        :return: A model name as a string.
        """
        return self._model_name

    def set_output_to(self, output_to: PathOrString) -> None:
        """
        Adjusts the file system settings in the present object such that all outputs are written to the given folder.
        :param output_to: The absolute path to a folder that should contain the outputs.
        """
        if isinstance(output_to, Path):
            output_to = str(output_to)
        self.output_to = output_to
        self.create_filesystem()

    def create_filesystem(self, project_root: Path = fixed_paths.repository_root_directory()) -> None:
        """
        Creates new file system settings (outputs folder, logs folder) based on the information stored in the
        present object. If any of the folders do not yet exist, they are created.
        :param project_root: The root folder for the codebase that triggers the training run.
        """
        self.file_system_config = DeepLearningFileSystemConfig.create(
            project_root=project_root,
            model_name=self.model_name,
            is_offline_run=is_offline_run_context(RUN_CONTEXT),
            output_to=self.output_to
        )

    @property
    def outputs_folder(self) -> Path:
        """Gets the full path in which the model outputs should be stored."""
        return self.file_system_config.outputs_folder

    @property
    def logs_folder(self) -> Path:
        """Gets the full path in which the model logs should be stored."""
        return self.file_system_config.logs_folder

    @property
    def checkpoint_folder(self) -> Path:
        """Gets the full path in which the model checkpoints should be stored during training."""
        return self.outputs_folder / CHECKPOINT_FOLDER

    @property
    def visualization_folder(self) -> Path:
        """Gets the full path in which the visualizations notebooks should be saved during training."""
        return self.outputs_folder / VISUALIZATION_FOLDER

    def get_path_to_checkpoint(self) -> Path:
        """
        Returns the full path to a recovery checkpoint.
        """
        return get_recovery_checkpoint_path(self.checkpoint_folder)

    def get_path_to_best_checkpoint(self) -> Path:
        """
        Returns the full path to a checkpoint file that was found to be best during training, whatever criterion
        was applied there.
        """
        return get_best_checkpoint_path(self.checkpoint_folder)


class OptimizerParams(param.Parameterized):
    l_rate: float = param.Number(1e-4, doc="The initial learning rate", bounds=(0, None))
    _min_l_rate: float = param.Number(0.0, doc="The minimum learning rate for the Polynomial and Cosine schedulers.",
                                      bounds=(0.0, None))
    l_rate_scheduler: LRSchedulerType = param.ClassSelector(default=LRSchedulerType.Polynomial,
                                                            class_=LRSchedulerType,
                                                            instantiate=False,
                                                            doc="Learning rate decay method (Cosine, Polynomial, "
                                                                "Step, MultiStep or Exponential)")
    l_rate_exponential_gamma: float = param.Number(0.9, doc="Controls the rate of decay for the Exponential "
                                                            "LR scheduler.")
    l_rate_step_gamma: float = param.Number(0.1, doc="Controls the rate of decay for the "
                                                     "Step LR scheduler.")
    l_rate_step_step_size: int = param.Integer(50, bounds=(0, None),
                                               doc="The step size for Step LR scheduler")
    l_rate_multi_step_gamma: float = param.Number(0.1, doc="Controls the rate of decay for the "
                                                           "MultiStep LR scheduler.")
    l_rate_multi_step_milestones: Optional[List[int]] = param.List(None, bounds=(1, None),
                                                                   allow_None=True, class_=int,
                                                                   doc="The milestones for MultiStep decay.")
    l_rate_polynomial_gamma: float = param.Number(1e-4, doc="Controls the rate of decay for the "
                                                            "Polynomial LR scheduler.")
    l_rate_warmup: LRWarmUpType = param.ClassSelector(default=LRWarmUpType.NoWarmUp, class_=LRWarmUpType,
                                                      instantiate=False,
                                                      doc="The type of learning rate warm up to use. "
                                                          "Can be NoWarmUp (default) or Linear.")
    l_rate_warmup_epochs: int = param.Integer(0, bounds=(0, None),
                                              doc="Number of warmup epochs (linear warmup) before the "
                                                  "scheduler starts decaying the learning rate. "
                                                  "For example, if you are using MultiStepLR with "
                                                  "milestones [50, 100, 200] and warmup epochs = 100, warmup "
                                                  "will last for 100 epochs and the first decay of LR "
                                                  "will happen on epoch 150")
    optimizer_type: OptimizerType = param.ClassSelector(default=OptimizerType.Adam, class_=OptimizerType,
                                                        instantiate=False, doc="The optimizer_type to use")
    opt_eps: float = param.Number(1e-4, doc="The epsilon parameter of RMSprop or Adam")
    rms_alpha: float = param.Number(0.9, doc="The alpha parameter of RMSprop")
    adam_betas: TupleFloat2 = param.NumericTuple((0.9, 0.999), length=2,
                                                 doc="The betas parameter of Adam, default is (0.9, 0.999)")
    momentum: float = param.Number(0.6, doc="The momentum parameter of the optimizers")
    weight_decay: float = param.Number(1e-4, doc="The weight decay used to control L2 regularization")

    def validate(self) -> None:
        if len(self.adam_betas) < 2:
            raise ValueError(
                "The adam_betas parameter should be the coefficients used for computing running averages of "
                "gradient and its square")

        if self.l_rate_scheduler == LRSchedulerType.MultiStep:
            if not self.l_rate_multi_step_milestones:
                raise ValueError("Must specify l_rate_multi_step_milestones to use LR scheduler MultiStep")
            if sorted(set(self.l_rate_multi_step_milestones)) != self.l_rate_multi_step_milestones:
                raise ValueError("l_rate_multi_step_milestones must be a strictly increasing list")
            if self.l_rate_multi_step_milestones[0] <= 0:
                raise ValueError("l_rate_multi_step_milestones cannot be negative or 0.")

    @property
    def min_l_rate(self) -> float:
        return self._min_l_rate

    @min_l_rate.setter
    def min_l_rate(self, value: float) -> None:
        if value > self.l_rate:
            raise ValueError("l_rate must be >= min_l_rate, found: {}, {}".format(self.l_rate, value))
        self._min_l_rate = value


class TrainerParams(param.Parameterized):
    num_epochs: int = param.Integer(100, bounds=(1, None), doc="Number of epochs to train.")
    recovery_checkpoint_save_interval: int = param.Integer(10, bounds=(0, None),
                                                           doc="Save epoch checkpoints when epoch number is a multiple "
                                                               "of recovery_checkpoint_save_interval. The intended use "
                                                               "is to allow restore training from failed runs.")
    recovery_checkpoints_save_last_k: int = param.Integer(default=1, bounds=(-1, None),
                                                          doc="Number of recovery checkpoints to keep. Recovery "
                                                              "checkpoints will be stored as recovery_epoch:{"
                                                              "epoch}.ckpt. If set to -1 keep all recovery "
                                                              "checkpoints.")
    detect_anomaly: bool = param.Boolean(False, doc="If true, test gradients for anomalies (NaN or Inf) during "
                                                    "training.")
    use_mixed_precision: bool = param.Boolean(False, doc="If true, mixed precision training is activated during "
                                                         "training.")
    max_num_gpus: int = param.Integer(default=-1, doc="The maximum number of GPUS to use. If set to a value < 0, use"
                                                      "all available GPUs. In distributed training, this is the "
                                                      "maximum number of GPUs per node.")
    pl_progress_bar_refresh_rate: Optional[int] = \
        param.Integer(default=None,
                      doc="PyTorch Lightning trainer flag 'progress_bar_refresh_rate': How often to refresh progress "
                          "bar (in steps). Value 0 disables progress bar. Value None chooses automatically.")
    pl_num_sanity_val_steps: int = \
        param.Integer(default=0,
                      doc="PyTorch Lightning trainer flag 'num_sanity_val_steps': Number of validation "
                          "steps to run before training, to identify possible problems")
    pl_deterministic: bool = \
        param.Integer(default=True,
                      doc="Controls the PyTorch Lightning trainer flags 'deterministic' and 'benchmark'. If "
                          "'pl_deterministic' is True, results are perfectly reproducible. If False, they are not, but "
                          "you may see training speed increases.")
    pl_find_unused_parameters: bool = \
        param.Boolean(default=False,
                      doc="Controls the PyTorch Lightning flag 'find_unused_parameters' for the DDP plugin. "
                          "Setting it to True comes with a performance hit.")

    @property
    def use_gpu(self) -> bool:
        """
        Returns True if a GPU is available, and the self.max_num_gpus flag allows it to be used. Returns False
        otherwise (i.e., if there is no GPU available, or self.max_num_gpus==0)
        """
        if self.max_num_gpus == 0:
            return False
        from InnerEye.ML.utils.ml_util import is_gpu_available
        return is_gpu_available()

    @property
    def num_gpus_per_node(self) -> int:
        """
        Computes the number of gpus to use for each node: either the number of gpus available on the device
        or restrict it to max_num_gpu, whichever is smaller. Returns 0 if running on a CPU device.
        """
        import torch
        num_gpus = torch.cuda.device_count() if self.use_gpu else 0
        logging.info(f"Number of available GPUs: {num_gpus}")
        if 0 <= self.max_num_gpus < num_gpus:
            num_gpus = self.max_num_gpus
            logging.info(f"Restricting the number of GPUs to {num_gpus}")
        elif self.max_num_gpus > num_gpus:
            logging.warning(f"You requested max_num_gpus {self.max_num_gpus} but there are only {num_gpus} available.")
        return num_gpus


class DeepLearningConfig(WorkflowParams,
                         DatasetParams,
                         OutputParams,
                         OptimizerParams,
                         TrainerParams,
                         GenericConfig):
    """
    A class that holds all settings that are shared across segmentation models and regression/classification models.
    """
    _model_category: ModelCategory = param.ClassSelector(class_=ModelCategory,
                                                         doc="The high-level model category described by this config.")

    num_dataload_workers: int = param.Integer(2, bounds=(0, None),
                                              doc="The number of data loading workers (processes). When set to 0,"
                                                  "data loading is running in the same process (no process startup "
                                                  "cost, hence good for use in unit testing. However, it "
                                                  "does not give the same result as running with 1 worker process)")
    shuffle: bool = param.Boolean(True, doc="If true, the dataset will be shuffled randomly during training.")
    train_batch_size: int = param.Integer(4, bounds=(0, None),
                                          doc="The number of crops that make up one minibatch during training.")
    use_model_parallel: bool = param.Boolean(False, doc="If true, neural network model is partitioned across all "
                                                        "available GPUs to fit in a large model. It shall not be used "
                                                        "together with data parallel.")
    pin_memory: bool = param.Boolean(True, doc="Value of pin_memory argument to DataLoader")
    restrict_subjects: Optional[str] = \
        param.String(doc="Use at most this number of subjects for train, val, or test set (must be > 0 or None). "
                         "If None, do not modify the train, val, or test sets. If a string of the form 'i,j,k' where "
                         "i, j and k are integers, modify just the corresponding sets (i for train, j for val, k for "
                         "test). If any of i, j or j are missing or are negative, do not modify the corresponding "
                         "set. Thus a value of 20,,5 means limit training set to 20, keep validation set as is, and "
                         "limit test set to 5. If any of i,j,k is '+', discarded members of the other sets are added "
                         "to that set.",
                     allow_None=True)
    _dataset_data_frame: Optional[DataFrame] = \
        param.DataFrame(default=None,
                        doc="The dataframe that contains the dataset for the model. This is usually read from disk "
                            "from dataset.csv")
    avoid_process_spawn_in_data_loaders: bool = \
        param.Boolean(is_windows(), doc="If True, use a data loader logic that avoid spawning new processes at the "
                                        "start of each epoch. This speeds up training on both Windows and Linux, but"
                                        "on Linux, inference is currently disabled as the data loaders hang. "
                                        "If False, use the default data loader logic that starts new processes for "
                                        "each epoch.")
    max_batch_grad_cam: int = param.Integer(default=0, doc="Max number of validation batches for which "
                                                           "to save gradCam images. By default "
                                                           "visualizations are saved for all images "
                                                           "in the validation set")
    label_smoothing_eps: float = param.Number(0.0, bounds=(0.0, 1.0),
                                              doc="Target smoothing value for label smoothing")
    log_to_parent_run: bool = param.Boolean(default=False, doc="If true, hyperdrive child runs will log their metrics"
                                                               "to their parent run.")
    use_imbalanced_sampler_for_training: bool = param.Boolean(default=False,
                                                              doc="If True, use an imbalanced sampler during training.")
    drop_last_batch_in_training: bool = param.Boolean(default=False,
                                                      doc="If True, drop the last incomplete batch during"
                                                          "training. If all batches are complete, no batch gets "
                                                          "dropped. If False, keep all batches.")
    log_summaries_to_files: bool = param.Boolean(
        default=True,
        doc="If True, model summaries are logged to files in logs/model_summaries; "
            "if False, to stdout or driver log")
    mean_teacher_alpha: float = param.Number(bounds=(0, 1), allow_None=True, default=None,
                                             doc="If this value is set, the mean teacher model will be computed. "
                                                 "Currently only supported for scalar models. In this case, we only "
                                                 "report metrics and cross-validation results for "
                                                 "the mean teacher model. Likewise the model used for inference "
                                                 "is the mean teacher model. The student model is only used for "
                                                 "training. Alpha is the momentum term for weight updates of the mean "
                                                 "teacher model. After each training step the mean teacher model "
                                                 "weights are updated using mean_teacher_"
                                                 "weight = alpha * (mean_teacher_weight) "
                                                 " + (1-alpha) * (current_student_weights). ")
    #: Name of the csv file providing information on the dataset to be used.
    dataset_csv: str = param.String(
        DATASET_CSV_FILE_NAME,
        doc="Name of the CSV file providing information on the dataset to be used. "
            "For segmentation models, this file must contain at least the fields: `subject`, `channel`, `filePath`.")

    def __init__(self, **params: Any) -> None:
        self._model_name = type(self).__name__
        # This should be annotated as torch.utils.data.Dataset, but we don't want to import torch here.
        self._datasets_for_training: Optional[Dict[ModelExecutionMode, Any]] = None
        self._datasets_for_inference: Optional[Dict[ModelExecutionMode, Any]] = None
        self.recovery_start_epoch = 0
        super().__init__(throw_if_unknown_param=True, **params)
        logging.info("Creating the default output folder structure.")
        self.create_filesystem(fixed_paths.repository_root_directory())
        # Disable the PL progress bar because all InnerEye models have their own console output
        self.pl_progress_bar_refresh_rate = 0
        self.extra_downloaded_run_id: Optional[Any] = None

    def validate(self) -> None:
        """
        Validates the parameters stored in the present object.
        """
        WorkflowParams.validate(self)
        OptimizerParams.validate(self)
        DatasetParams.validate(self)

    @property
    def model_category(self) -> ModelCategory:
        """
        Gets the high-level model category that this configuration objects represents (segmentation or scalar output).
        """
        return self._model_category

    @property
    def is_segmentation_model(self) -> bool:
        """
        Returns True if the present model configuration belongs to the high-level category ModelCategory.Segmentation.
        """
        return self.model_category == ModelCategory.Segmentation

    @property
    def is_scalar_model(self) -> bool:
        """
        Returns True if the present model configuration belongs to the high-level category ModelCategory.Scalar
        i.e. for Classification or Regression models.
        """
        return self.model_category.is_scalar

    @property
    def compute_grad_cam(self) -> bool:
        return self.max_batch_grad_cam > 0

    @property
    def dataset_data_frame(self) -> Optional[DataFrame]:
        """
        Gets the pandas data frame that the model uses.
        :return:
        """
        return self._dataset_data_frame

    @dataset_data_frame.setter
    def dataset_data_frame(self, data_frame: Optional[DataFrame]) -> None:
        """
        Sets the pandas data frame that the model uses.
        :param data_frame: The data frame to set.
        """
        self._dataset_data_frame = data_frame

    def get_train_epochs(self) -> List[int]:
        """
        Returns the epochs for which training will be performed.
        :return:
        """
        return list(range(self.recovery_start_epoch + 1, self.num_epochs + 1))

    def get_total_number_of_training_epochs(self) -> int:
        """
        Returns the number of epochs for which a model will be trained.
        :return:
        """
        return len(self.get_train_epochs())

    def get_total_number_of_validation_epochs(self) -> int:
        """
        Returns the number of epochs for which a model will be validated.
        :return:
        """
        return self.get_total_number_of_training_epochs()

    @property
    def compute_mean_teacher_model(self) -> bool:
        """
        Returns True if the mean teacher model should be computed.
        """
        return self.mean_teacher_alpha is not None

    def __str__(self) -> str:
        """Returns a string describing the present object, as a list of key: value strings."""
        arguments_str = "\nArguments:\n"
        # Avoid callable params, the bindings that are printed out can be humongous.
        # Avoid dataframes
        skip_params = {name for name, value in self.param.params().items()
                       if isinstance(value, (param.Callable, param.DataFrame))}
        for key, value in self.param.get_param_values():
            if key not in skip_params:
                arguments_str += f"\t{key:40}: {value}\n"
        return arguments_str

    def load_checkpoint_and_modify(self, path_to_checkpoint: Path) -> Dict[str, Any]:
        """
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


def load_checkpoint(path_to_checkpoint: Path, use_gpu: bool = True) -> Dict[str, Any]:
    """
    Loads a Torch checkpoint from the given file. If use_gpu==False, map all parameters to the GPU, otherwise
    left the device of all parameters unchanged.
    """
    import torch
    map_location = None if use_gpu else 'cpu'
    checkpoint = torch.load(str(path_to_checkpoint), map_location=map_location)
    return checkpoint

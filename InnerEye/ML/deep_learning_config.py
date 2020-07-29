#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from enum import Enum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional

import param
from pandas import DataFrame
from param import Parameterized

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import MetricsDataframeLoggers, is_windows
from InnerEye.Common.fixed_paths import DEFAULT_LOGS_DIR_NAME, DEFAULT_AML_UPLOAD_DIR
from InnerEye.Common.generic_parsing import CudaAwareConfig, GenericConfig
from InnerEye.Common.type_annotations import PathOrString, TupleFloat2
from InnerEye.ML.common import CHECKPOINT_FILE_SUFFIX, ModelExecutionMode, create_unique_timestamp_id

VISUALIZATION_FOLDER = "Visualizations"
CHECKPOINT_FOLDER = "checkpoints"
ARGS_TXT = "args.txt"


@unique
class LRSchedulerType(Enum):
    """
    Supported lr scheduler types for model training
    """
    Exponential = "Exponential"
    Step = "Step"
    Polynomial = "Polynomial"
    Cosine = "Cosine"


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
    Scalar = "Scalar"  # All models that predict a scalar (classification, regression) from an input image.


@unique
class MultiprocessingStartMethod(Enum):
    """
    Different methods for starting data loader processes.
    """
    fork = "fork"
    forkserver = "forkserver"
    spawn = "spawn"


class DeepLearningFileSystemConfig(Parameterized):
    """High level config to abstract the file system related configs for deep learning models"""
    outputs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                               doc="The folder where all training and test outputs should go.")
    logs_folder: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                            doc="The folder for all log files and Tensorboard event files")
    project_root: Path = param.ClassSelector(class_=Path, default=Path(), instantiate=False,
                                             doc="The root folder for the codebase that triggers the training run.")
    run_folder: Optional[Path] = param.ClassSelector(class_=Path, default=None, instantiate=False,
                                                     doc="For runs outside AzureML, this is the folder that contains "
                                                         "outputs and the logs subfolder.")

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

        run_folder: Optional[Path]
        if is_offline_run:
            logging.info("Running outside of AzureML.")
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
            run_folder = None
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
        raise ValueError("This method should only be called for offline runs, when the logs folder is inside the "
                         "outputs folder.")


class DeepLearningConfig(GenericConfig, CudaAwareConfig):
    """
    A class that holds all settings that are shared across segmentation models and regression/classification models.
    """
    _model_category: ModelCategory = param.ClassSelector(class_=ModelCategory,
                                                         doc="The high-level model category described by this config.")
    _model_name: str = param.String(None, doc="The human readable name of the model (for example, Liver). This is "
                                              "usually set from the class name.")
    random_seed: int = param.Integer(42, doc="The seed to use for all random number generators.")
    azure_dataset_id: Optional[str] = param.String(None, allow_None=True,
                                                   doc="The ID of the dataset to use. This dataset must exist as a "
                                                       "folder of the same name in the 'datasets' "
                                                       "container in the datasets storage account.")
    local_dataset: Optional[Path] = param.ClassSelector(class_=Path,
                                                        default=None,
                                                        allow_None=True,
                                                        doc="The path of the dataset to use, when training is running "
                                                            "outside Azure.")
    num_dataload_workers: int = param.Integer(8, bounds=(0, None),
                                              doc="The number of data loading workers (processes). When set to 0,"
                                                  "data loading is running in the same process (no process startup "
                                                  "cost, hence good for use in unit testing. However, it "
                                                  "does not give the same result as running with 1 worker process)")
    shuffle: bool = param.Boolean(True, doc="If true, the dataset will be shuffled randomly during training.")
    num_epochs: int = param.Integer(100, bounds=(1, None), doc="Number of epochs to train.")
    start_epoch: int = param.Integer(0, bounds=(0, None), doc="The first epoch to train. Set to 0 to start a new "
                                                              "training. Set to a value larger than zero for starting"
                                                              " from a checkpoint.")

    l_rate: float = param.Number(1e-4, doc="The initial learning rate", bounds=(0, None))
    _min_l_rate: float = param.Number(0.0, doc="The minimum learning rate", bounds=(0.0, None))
    l_rate_decay: LRSchedulerType = param.ClassSelector(default=LRSchedulerType.Polynomial,
                                                        class_=LRSchedulerType,
                                                        instantiate=False,
                                                        doc="Learning rate decay method (Cosine, Polynomial, "
                                                            "Step, or Exponential)")
    l_rate_gamma: float = param.Number(1e-4, doc="Controls the rate of decay, depending on the method")
    l_rate_step_size: int = param.Integer(50, bounds=(0, None),
                                          doc="The step size for Step decay (ignored for Polynomial "
                                              "and Exponential)")

    optimizer_type: OptimizerType = param.ClassSelector(default=OptimizerType.Adam, class_=OptimizerType,
                                                        instantiate=False, doc="The optimizer_type to use")
    opt_eps: float = param.Number(1e-4, doc="The epsilon parameter of RMSprop or Adam")
    rms_alpha: float = param.Number(0.9, doc="The alpha parameter of RMSprop")
    adam_betas: TupleFloat2 = param.NumericTuple((0.9, 0.999), length=2,
                                                 doc="The betas parameter of Adam, default is (0.9, 0.999)")
    momentum: float = param.Number(0.6, doc="The momentum parameter of the optimizers")
    weight_decay: float = param.Number(1e-4, doc="The weight decay used to control L2 regularization")

    save_start_epoch: int = param.Integer(100, bounds=(0, None), doc="Save epoch checkpoints only when epoch is "
                                                                     "larger or equal to this value.")
    save_step_epochs: int = param.Integer(50, bounds=(0, None), doc="Save epoch checkpoints when epoch number is a "
                                                                    "multiple of save_step_epochs")
    train_batch_size: int = param.Integer(4, bounds=(0, None),
                                          doc="The number of crops that make up one minibatch during training.")
    detect_anomaly: bool = param.Boolean(False, doc="If true, test gradients for anomalies (NaN or Inf) during "
                                                    "training.")
    use_mixed_precision: bool = param.Boolean(False, doc="If true, mixed precision training is activated during "
                                                         "training.")
    use_model_parallel: bool = param.Boolean(False, doc="If true, neural network model is partitioned across all "
                                                        "available GPUs to fit in a large model. It shall not be used "
                                                        "together with data parallel.")
    test_diff_epochs: Optional[int] = param.Integer(None, doc="Number of different epochs of the same model to test",
                                                    allow_None=True)
    test_step_epochs: Optional[int] = param.Integer(None, doc="How many epochs to move for each test", allow_None=True)
    test_start_epoch: Optional[int] = param.Integer(None, doc="The first epoch on which testing should run.",
                                                    allow_None=True)
    monitoring_interval_seconds: int = param.Integer(0, doc="Seconds delay between logging GPU/CPU resource "
                                                            "statistics. If 0 or less, do not log any resource "
                                                            "statistics.")
    number_of_cross_validation_splits: int = param.Integer(0, bounds=(0, None),
                                                           doc="Number of cross validation splits for k-fold cross "
                                                               "validation")
    cross_validation_split_index: int = param.Integer(-1, bounds=(-1, None),
                                                      doc="The index of the cross validation fold this model is "
                                                          "associated with when performing k-fold cross validation")
    file_system_config: DeepLearningFileSystemConfig = param.ClassSelector(default=DeepLearningFileSystemConfig(),
                                                                           class_=DeepLearningFileSystemConfig,
                                                                           instantiate=False,
                                                                           doc="File system related configs")
    pin_memory: bool = param.Boolean(True, doc="Value of pin_memory argument to DataLoader")
    _overrides: Dict[str, Any] = param.Dict(instantiate=True,
                                            doc="Model config properties that were overridden from the commandline")
    restrict_subjects: Optional[str] = \
        param.String(doc="Use at most this number of subjects for train, val, or test set (must be > 0 or None). "
                         "If None, do not modify the train, val, or test sets. If a string of the form 'i,j,k' where "
                         "i, j and k are integers, modify just the corresponding set (i for train, j for val, k for "
                         "test). If any of i, j or j are missing or are non-positive, do not modify the corresponding "
                         "set. Thus a value of 20,,5 means limit training set to 20, keep validation set as is, and "
                         "limit test set to 5.",
                     allow_None=True)
    perform_training_set_inference: bool = \
        param.Boolean(False,
                      doc="If False (default), run full image inference on validation and test set after training. If "
                          "True, also run full image inference on the training set")
    perform_validation_and_test_set_inference: bool = \
        param.Boolean(True,
                      doc="If True (default), run full image inference on validation and test set after training.")
    _metrics_data_frame_loggers: MetricsDataframeLoggers = param.ClassSelector(default=None,
                                                                               class_=MetricsDataframeLoggers,
                                                                               instantiate=False,
                                                                               doc="Data frame loggers for this model "
                                                                                   "config")
    _dataset_data_frame: Optional[DataFrame] = \
        param.DataFrame(default=None,
                        doc="The dataframe that contains the dataset for the model. This is usually read from disk "
                            "from dataset.csv")
    _use_gpu: Optional[bool] = param.Boolean(None,
                                             doc="If true, a CUDA capable GPU with at least 1 device is "
                                                 "available. If None, the use_gpu property has not yet been called.")
    avoid_process_spawn_in_data_loaders: bool = \
        param.Boolean(is_windows(), doc="If True, use a data loader logic that avoid spawning new processes at the "
                                        "start of each epoch. This speeds up training on both Windows and Linux, but"
                                        "on Linux, inference is currently disabled as the data loaders hang. "
                                        "If False, use the default data loader logic that starts new processes for "
                                        "each epoch.")
    # The default multiprocessing start_method in both PyTorch and the Python standardlibrary is "fork" for Linux and
    # "spawn" (the only available method) for Windows. However if the user does not specify a method, we default to
    # "forkserver" for Linux, as there is evidence that this greatly reduces the risk of dataloader processes getting
    # stuck.
    multiprocessing_start_method: MultiprocessingStartMethod = \
        param.ClassSelector(class_=MultiprocessingStartMethod,
                            default=(MultiprocessingStartMethod.spawn if is_windows()
                                     else MultiprocessingStartMethod.forkserver),
                            doc="Method to be used to start child processes in pytorch. Should be one of forkserver, "
                                "fork or spawn. If not specified, forkserver is used on Linux and spawn on Windows.")
    output_to: Optional[str] = \
        param.String(default=None,
                     doc="If provided, the run outputs will be written to the given folder. If not provided, outputs "
                         "will go into a subfolder of the project root folder.")
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

    def __init__(self, **params: Any) -> None:
        self._model_name = type(self).__name__
        # This should be annotated as torch.utils.data.Dataset, but we don't want to import torch here.
        self._datasets_for_training: Optional[Dict[ModelExecutionMode, Any]] = None
        self._datasets_for_inference: Optional[Dict[ModelExecutionMode, Any]] = None
        super().__init__(**params)
        logging.info("Creating the default output folder structure.")
        self.create_filesystem(fixed_paths.repository_root_directory())

    def validate(self) -> None:
        """
        Validates the parameters stored in the present object.
        """
        if len(self.adam_betas) < 2:
            raise ValueError(
                "The adam_betas parameter should be the coefficients used for computing running averages of "
                "gradient and its square")

        if self.azure_dataset_id is None and self.local_dataset is None:
            raise ValueError("Either of local_dataset or azure_dataset_id must be set.")

        if 0 < self.number_of_cross_validation_splits <= self.cross_validation_split_index:
            raise ValueError(f"Cross validation split index is out of bounds: {self.cross_validation_split_index}, "
                             f"which is invalid for CV with {self.number_of_cross_validation_splits} splits.")
        elif self.number_of_cross_validation_splits == 0 and self.cross_validation_split_index != -1:
            raise ValueError(f"Cross validation split index must be -1 for a non cross validation run, "
                             f"found number_of_cross_validation_splits = {self.number_of_cross_validation_splits} "
                             f"and cross_validation_split_index={self.cross_validation_split_index}")

    @property
    def model_name(self) -> str:
        """
        Gets the human readable name of the model (e.g., Liver). This is usually set from the class name.
        :return: A model name as a string.
        """
        return self._model_name

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
        return self.model_category == ModelCategory.Scalar

    @property
    def compute_grad_cam(self) -> bool:
        return self.max_batch_grad_cam > 0

    @property
    def min_l_rate(self) -> float:
        return self._min_l_rate

    @min_l_rate.setter
    def min_l_rate(self, value: float) -> None:
        if value > self.l_rate:
            raise ValueError("l_rate must be >= min_l_rate, found: {}, {}".format(self.l_rate, value))
        self._min_l_rate = value

    @property
    def outputs_folder(self) -> Path:
        """Gets the full path in which the model outputs should be stored."""
        return self.file_system_config.outputs_folder

    @property
    def logs_folder(self) -> Path:
        """Gets the full path in which the model logs should be stored."""
        return self.file_system_config.logs_folder

    @property
    def checkpoint_folder(self) -> str:
        """Gets the full path in which the model checkpoints should be stored during training."""
        return str(self.outputs_folder / CHECKPOINT_FOLDER)

    @property
    def visualization_folder(self) -> Path:
        """Gets the full path in which the visualizations notebooks should be saved during training."""
        return self.outputs_folder / VISUALIZATION_FOLDER

    @property
    def perform_cross_validation(self) -> bool:
        """
        True if cross validation will be be performed as part of the training procedure.
        :return:
        """
        return self.number_of_cross_validation_splits > 0

    @property
    def overrides(self) -> Optional[Dict[str, Any]]:
        return self._overrides

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

    @property
    def metrics_data_frame_loggers(self) -> MetricsDataframeLoggers:
        """
        Gets the metrics data frame loggers for this config.
        :return:
        """
        return self._metrics_data_frame_loggers

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
            is_offline_run=self.is_offline_run,
            output_to=self.output_to
        )

    def create_dataframe_loggers(self) -> None:
        """
        Initializes the metrics loggers that are stored in self._metrics_data_frame_loggers
        :return:
        """
        self._metrics_data_frame_loggers = MetricsDataframeLoggers(outputs_folder=self.outputs_folder)

    def should_load_checkpoint_for_training(self) -> bool:
        """Returns true if start epoch > 0, that is, if an existing checkpoint is used to continue training."""
        return self.start_epoch > 0

    def should_save_epoch(self, epoch: int) -> bool:
        """Returns True if the present epoch should be saved, as per the save_start_epoch and save_step_epochs
        settings. Epoch writing starts with the first epoch that is >= save_start_epoch, and that
        is evenly divisible by save_step_epochs. A checkpoint is always written for the last epoch (num_epochs),
        such that it is easy to overwrite num_epochs on the commandline without having to change the test parameters
        at the same time.
        :param epoch: The current epoch. The first epoch is assumed to be 1."""
        should_save_epoch = epoch >= self.save_start_epoch \
                            and epoch % self.save_step_epochs == 0
        is_last_epoch = epoch == self.num_epochs
        return should_save_epoch or is_last_epoch

    def get_test_epochs(self) -> List[int]:
        """
        Returns the list of epochs for which the model should be evaluated on full images in the test set.
        These are all epochs starting at self.test_start_epoch, in intervals of self.n_steps_epoch.
        The last training epoch is always included. If either of the self.test_* fields is missing (set to None),
        only the last training epoch is returned.
        :return:
        """
        test_epochs = {self.num_epochs}
        if self.test_diff_epochs is not None and self.test_start_epoch is not None and \
                self.test_step_epochs is not None:
            for j in range(self.test_diff_epochs):
                epoch = self.test_start_epoch + self.test_step_epochs * j
                if epoch > self.num_epochs:
                    break
                test_epochs.add(epoch)
        return sorted(test_epochs)

    def get_path_to_checkpoint(self, epoch: int) -> Path:
        """
        Returns full path to a checkpoint given an epoch
        :param epoch: the epoch number
        :return: path to a checkpoint given an epoch
        """
        return fixed_paths.repository_root_directory() \
               / self.checkpoint_folder \
               / f"{epoch}{CHECKPOINT_FILE_SUFFIX}"

    @property  # type: ignore
    def use_gpu(self) -> bool:  # type: ignore
        """
        Returns True if a CUDA capable GPU is present and should be used, False otherwise.
        """
        if self._use_gpu is None:
            # Use a local import here because we don't want the whole file to depend on pytorch.
            from InnerEye.ML.utils.ml_util import is_gpu_available
            self._use_gpu = is_gpu_available()
        return self._use_gpu

    @use_gpu.setter
    def use_gpu(self, value: bool) -> None:
        """
        Sets the flag that controls the use of the GPU. Raises a ValueError if the value is True, but no GPU is
        present.
        """
        if value:
            # Use a local import here because we don't want the whole file to depend on pytorch.
            from InnerEye.ML.utils.ml_util import is_gpu_available
            if not is_gpu_available():
                raise ValueError("Can't set use_gpu to True if there is not CUDA capable GPU present.")
        self._use_gpu = value

    @property
    def use_data_parallel(self) -> bool:
        """
        Data parallel is used if GPUs are usable and the number of CUDA devices are greater than 1.
        :return:
        """
        _devices = self.get_cuda_devices()
        return _devices is not None and len(_devices) > 1

    def write_args_file(self, root: Optional[Path] = None) -> None:
        """
        Writes the current config to disk. The file is written either to the given folder, or if omitted,
        to the default outputs folder.
        """
        dst = (root or self.outputs_folder) / ARGS_TXT
        dst.write_text(data=str(self))

    @property
    def should_wait_for_other_cross_val_child_runs(self) -> bool:
        """
        Returns True if the current run is an online run and is the 0th cross validation split.
        In this case, this will be the run that will wait for all other child runs to finish in order
        to aggregate their results.
        :return:
        """
        return (not self.is_offline_run) and self.cross_validation_split_index == 0

    @property
    def is_offline_run(self) -> bool:
        """
        Returns True if the run is executing outside AzureML, or False if inside AzureML.
        """
        return is_offline_run_context(RUN_CONTEXT)

    def __str__(self) -> str:
        """Returns a string describing the present object, as a list of key == value pairs."""
        arguments_str = "\nArguments:\n"
        property_dict = vars(self)
        keys = sorted(property_dict)
        for key in keys:
            arguments_str += "\t{:18}: {}\n".format(key, property_dict[key])
        return arguments_str

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple, TypeVar

from pytorch_lightning import Callback, Trainer, seed_everything
from pytorch_lightning.callbacks import GPUStatsMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin

from InnerEye.Azure.azure_runner import ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK
from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.common_util import SUBJECT_METRICS_FILE_NAME, change_working_directory
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import CombinedDataModule
from InnerEye.ML.common import ARGS_TXT, AUTOSAVE_CHECKPOINT_FILE_NAME, ModelExecutionMode, \
    VISUALIZATION_FOLDER
from InnerEye.ML.lightning_base import InnerEyeContainer, InnerEyeLightning
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.lightning_loggers import StoringLogger
from InnerEye.ML.lightning_models import SUBJECT_OUTPUT_PER_RANK_PREFIX, ScalarLightning, \
    get_subject_output_file_per_rank
from InnerEye.ML.utils.checkpoint_handling import cleanup_checkpoints
from health_azure.utils import is_global_rank_zero, is_local_rank_zero
from health_ml.utils import AzureMLLogger, AzureMLProgressBar

TEMP_PREFIX = "temp/"

T = TypeVar('T')


def upload_output_file_as_temp(file_path: Path, outputs_folder: Path) -> None:
    """
    Uploads a file to the AzureML run. It will get a name that is composed of a "temp/" prefix, plus the path
    of the file relative to the outputs folder that is used for training.

    :param file_path: The path of the file to upload.
    :param outputs_folder: The root folder that contains all training outputs.
    """
    upload_name = TEMP_PREFIX + str(file_path.relative_to(outputs_folder))
    RUN_CONTEXT.upload_file(upload_name, path_or_stream=str(file_path))


def write_args_file(config: Any, outputs_folder: Path) -> None:
    """
    Writes the given config to disk in plain text in the default output folder.
    """
    output = str(config)
    outputs_folder.mkdir(exist_ok=True, parents=True)
    dst = outputs_folder / ARGS_TXT
    dst.write_text(output)
    logging.info(output)


def create_lightning_trainer(container: LightningContainer,
                             resume_from_checkpoint: Optional[Path] = None,
                             num_nodes: int = 1,
                             multiple_trainloader_mode: str = "max_size_cycle") -> \
        Tuple[Trainer, StoringLogger]:
    """
    Creates a Pytorch Lightning Trainer object for the given model configuration. It creates checkpoint handlers
    and loggers. That includes a diagnostic logger for use in unit tests, that is also returned as the second
    return value.

    :param container: The container with model and data.
    :param resume_from_checkpoint: If provided, training resumes from this checkpoint point.
    :param num_nodes: The number of nodes to use in distributed training.
    :return: A tuple [Trainer object, diagnostic logger]
    """
    logging.debug(f"resume_from_checkpoint: {resume_from_checkpoint}")
    num_gpus = container.num_gpus_per_node()
    effective_num_gpus = num_gpus * num_nodes
    strategy = None
    if effective_num_gpus == 0:
        accelerator = "cpu"
        devices = 1
        message = "CPU"
    else:
        accelerator = "gpu"
        devices = num_gpus
        message = f"{devices} GPU"
        if effective_num_gpus > 1:
            # Accelerator should be "ddp" when running large models in AzureML (when using DDP_spawn, we get out of
            # GPU memory).
            # Initialize the DDP plugin. The default for pl_find_unused_parameters is False. If True, the plugin
            # prints out lengthy warnings about the performance impact of find_unused_parameters.
            strategy = DDPPlugin(find_unused_parameters=container.pl_find_unused_parameters)
            message += "s per node with DDP"
    logging.info(f"Using {message}")
    tensorboard_logger = TensorBoardLogger(save_dir=str(container.logs_folder), name="Lightning", version="")
    loggers = [tensorboard_logger, AzureMLLogger(False)]
    storing_logger = StoringLogger()
    loggers.append(storing_logger)
    # Use 32bit precision when running on CPU. Otherwise, make it depend on use_mixed_precision flag.
    precision = 32 if num_gpus == 0 else 16 if container.use_mixed_precision else 32
    # The next two flags control the settings in torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    # https://pytorch.org/docs/stable/notes/randomness.html
    # Note that switching to deterministic models can have large performance downside.
    if container.pl_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True

    # The last checkpoint is considered the "best" checkpoint. For large segmentation
    # models, this still appears to be the best way of choosing them because validation loss on the relatively small
    # training patches is not stable enough. Going by the validation loss somehow works for the Prostate model, but
    # not for the HeadAndNeck model.
    # Note that "last" is somehow a misnomer, it should rather be "latest". There is a "last" checkpoint written in
    # every epoch. We could use that for recovery too, but it could happen that the job gets preempted right during
    # writing that file, and we would end up with an invalid file.
    last_checkpoint_callback = ModelCheckpoint(dirpath=str(container.checkpoint_folder),
                                               save_last=True,
                                               save_top_k=0)
    recovery_checkpoint_callback = ModelCheckpoint(dirpath=str(container.checkpoint_folder),
                                                   filename=AUTOSAVE_CHECKPOINT_FILE_NAME,
                                                   every_n_epochs=container.autosave_every_n_val_epochs,
                                                   save_last=False)
    callbacks: List[Callback] = [
        last_checkpoint_callback,
        recovery_checkpoint_callback,
    ]
    if container.monitor_loading:
        # TODO antonsc: Remove after fixing the callback.
        raise NotImplementedError("Monitoring batch loading times has been temporarily disabled.")
        # callbacks.append(BatchTimeCallback())
    if num_gpus > 0 and container.monitor_gpu:
        logging.info("Adding monitoring for GPU utilization")
        callbacks.append(GPUStatsMonitor(intra_step_time=True, inter_step_time=True))
    # Add the additional callbacks that were specified in get_trainer_arguments for LightningContainers
    additional_args = container.get_trainer_arguments()
    # Callbacks can be specified via the "callbacks" argument (the legacy behaviour) or the new get_callbacks method
    if "callbacks" in additional_args:
        more_callbacks = additional_args.pop("callbacks")
        if isinstance(more_callbacks, list):
            callbacks.extend(more_callbacks)  # type: ignore
        else:
            callbacks.append(more_callbacks)  # type: ignore
    callbacks.extend(container.get_callbacks())
    is_azureml_run = not is_offline_run_context(RUN_CONTEXT)
    progress_bar_refresh_rate = container.pl_progress_bar_refresh_rate
    if progress_bar_refresh_rate is None:
        progress_bar_refresh_rate = 50
        logging.info(f"The progress bar refresh rate is not set. Using a default of {progress_bar_refresh_rate}. "
                     f"To change, modify the pl_progress_bar_refresh_rate field of the container.")
    if is_azureml_run:
        callbacks.append(AzureMLProgressBar(refresh_rate=progress_bar_refresh_rate,
                                            write_to_logging_info=True,
                                            print_timestamp=False))
    else:
        callbacks.append(TQDMProgressBar(refresh_rate=progress_bar_refresh_rate))
    # Read out additional model-specific args here.
    # We probably want to keep essential ones like numgpu and logging.
    trainer = Trainer(default_root_dir=str(container.outputs_folder),
                      deterministic=deterministic,
                      benchmark=benchmark,
                      accelerator=accelerator,
                      strategy=strategy,
                      max_epochs=container.num_epochs,
                      # Both these arguments can be integers or floats. If integers, it is the number of batches.
                      # If float, it's the fraction of batches. We default to 1.0 (processing all batches).
                      limit_train_batches=container.pl_limit_train_batches or 1.0,
                      limit_val_batches=container.pl_limit_val_batches or 1.0,
                      num_sanity_val_steps=container.pl_num_sanity_val_steps,
                      check_val_every_n_epoch=container.pl_check_val_every_n_epoch,
                      callbacks=callbacks,
                      logger=loggers,
                      num_nodes=num_nodes,
                      devices=devices,
                      precision=precision,
                      sync_batchnorm=True,
                      detect_anomaly=container.detect_anomaly,
                      profiler=container.pl_profiler,
                      resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
                      multiple_trainloader_mode=multiple_trainloader_mode,
                      **additional_args)
    return trainer, storing_logger


def start_resource_monitor(config: LightningContainer) -> ResourceMonitor:
    # initialize and start GPU monitoring
    gpu_tensorboard = config.logs_folder / "gpu_utilization"
    # Result file in CSV format should NOT live in the logs folder, the streaming upload that is
    # used for this folder might corrupt the file.
    gpu_csv = config.outputs_folder / "gpu_utilization"
    gpu_csv.mkdir(parents=True, exist_ok=True)
    logging.info(f"Starting resource monitor. GPU utilization will be written to Tensorboard in "
                 f"{gpu_tensorboard}, aggregate metrics to {gpu_csv}")
    resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                       tensorboard_folder=gpu_tensorboard,
                                       csv_results_folder=gpu_csv)
    resource_monitor.start()
    return resource_monitor


def model_train(checkpoint_path: Optional[Path],
                container: LightningContainer,
                num_nodes: int = 1) -> Tuple[Trainer, StoringLogger]:
    """
    The main training loop. It creates the Pytorch model based on the configuration options passed in,
    creates a Pytorch Lightning trainer, and trains the model.
    If a checkpoint was specified, then it loads the checkpoint before resuming training.

    :param checkpoint_path: Checkpoint path for model initialization
    :param num_nodes: The number of nodes to use in distributed training.
    :param container: A container object that holds the training data in PyTorch Lightning format
        and the model to train.
    :return: A tuple of [Trainer, StoringLogger]. Trainer is the Lightning Trainer object that was used for fitting
        the model. The StoringLogger object is returned when training an InnerEye built-in model, this is None when
        fitting other models.
    """
    lightning_model = container.model

    resource_monitor: Optional[ResourceMonitor] = None
    # Execute some bookkeeping tasks only once if running distributed:
    if is_global_rank_zero():
        logging.info(f"Model checkpoints are saved at {container.checkpoint_folder}")
        write_args_file(container.config if isinstance(container, InnerEyeContainer) else container,
                        outputs_folder=container.outputs_folder)
        if container.monitoring_interval_seconds > 0:
            resource_monitor = start_resource_monitor(container)

    # Run all of the container-related operations consistently with changed outputs folder, even ones that
    # should not rely on the current working directory, like get_data_module.
    with change_working_directory(container.outputs_folder):
        data_module = container.get_data_module()
        if is_global_rank_zero():
            container.before_training_on_global_rank_zero()
        if is_local_rank_zero():
            container.before_training_on_local_rank_zero()
        container.before_training_on_all_ranks()

    # Workaround for a bug in PL 1.5.5: We need to pass the cycle mode for the training data as a trainer argument
    # because training data that uses a CombinedLoader is not split correctly in DDP
    multiple_trainloader_mode = "max_size_cycle"
    if isinstance(data_module, CombinedDataModule):
        data_module.prepare_data()
        assert data_module.train_loader_cycle_mode is not None, "This field should be computed during prepare_data"
        multiple_trainloader_mode = data_module.train_loader_cycle_mode

    # Create the trainer object. Backup the environment variables before doing that, in case we need to run a second
    # training in the unit tests.d
    old_environ = dict(os.environ)
    # Set random seeds just before training. For segmentation models, we have
    # something that changes the random seed in the before_training_on_rank_zero hook.
    seed_everything(container.get_effective_random_seed())
    trainer, storing_logger = create_lightning_trainer(container,
                                                       checkpoint_path,
                                                       num_nodes=num_nodes,
                                                       multiple_trainloader_mode=multiple_trainloader_mode)
    rank_info = ", ".join(f"{env}: {os.getenv(env)}"
                          for env in [ENV_GLOBAL_RANK, ENV_LOCAL_RANK, ENV_NODE_RANK])
    logging.info(f"Environment variables: {rank_info}. trainer.global_rank: {trainer.global_rank}")
    # InnerEye models use this logger for diagnostics
    if isinstance(lightning_model, InnerEyeLightning):
        if storing_logger is None:
            raise ValueError("InnerEye models require the storing_logger for diagnostics")
        lightning_model.storing_logger = storing_logger

    logging.info("Starting training")

    trainer.fit(lightning_model, datamodule=data_module)
    trainer.logger.close()  # type: ignore

    world_size = getattr(trainer, "world_size", 0)
    is_azureml_run = not is_offline_run_context(RUN_CONTEXT)
    # Per-subject model outputs for regression models are written per rank, and need to be aggregated here.
    # Each thread per rank will come here, and upload its files to the run outputs. Rank 0 will later download them.
    if is_azureml_run and world_size > 1 and isinstance(lightning_model, ScalarLightning):
        upload_output_file_as_temp(lightning_model.train_subject_outputs_logger.csv_path,  # type: ignore
                                   container.outputs_folder)
        upload_output_file_as_temp(lightning_model.val_subject_outputs_logger.csv_path,  # type: ignore
                                   container.outputs_folder)
    # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
    # We can now use the global_rank of the Lightining model, rather than environment variables, because DDP has set
    # all necessary properties.
    if lightning_model.global_rank != 0:
        logging.info(f"Terminating training thread with rank {lightning_model.global_rank}.")
        sys.exit()

    logging.info("Removing redundant checkpoint files.")
    cleanup_checkpoints(container.checkpoint_folder)
    # Lightning modifies a ton of environment variables. If we first run training and then the test suite,
    # those environment variables will mislead the training runs in the test suite, and make them crash.
    # Hence, restore the original environment after training.
    os.environ.clear()
    os.environ.update(old_environ)

    if world_size and isinstance(lightning_model, ScalarLightning):
        if is_azureml_run and world_size > 1:
            # In a DDP run on the local box, all ranks will write to local disk, hence no download needed.
            # In a multi-node DDP, each rank would upload to AzureML, and rank 0 will now download all results and
            # concatenate
            for rank in range(world_size):
                for mode in [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
                    file = mode.value + "/" + get_subject_output_file_per_rank(rank)
                    RUN_CONTEXT.download_file(name=TEMP_PREFIX + file, output_file_path=container.outputs_folder / file)
        # Concatenate all temporary file per execution mode
        aggregate_and_create_subject_metrics_file(container.outputs_folder)

    logging.info("Finished training")

    # Upload visualization directory to AML run context to be able to see it in the Azure UI.
    if isinstance(container, InnerEyeContainer):
        if container.config.max_batch_grad_cam > 0 and container.visualization_folder.exists():
            RUN_CONTEXT.upload_folder(name=VISUALIZATION_FOLDER, path=str(container.visualization_folder))

    if resource_monitor:
        logging.info("Shutting down the resource monitor process.")
        if is_azureml_run:
            for gpu_name, metrics_per_gpu in resource_monitor.read_aggregate_metrics().items():
                # Log as a table, with GPU being the first column
                RUN_CONTEXT.log_row("GPU utilization", GPU=gpu_name, **metrics_per_gpu)
        resource_monitor.kill()

    return trainer, storing_logger


def aggregate_and_create_subject_metrics_file(outputs_folder: Path) -> None:
    """
    This functions takes all the subject metrics file written by each GPU (one file per GPU) and aggregates them into
    one single metrics file. Results is saved in ``config.outputs_folder / mode.value / SUBJECT_METRICS_FILE_NAME``.
    This is done for the metrics files for training and for validation data separately.

    :param config: model config
    """
    for mode in [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
        temp_files = sorted((outputs_folder / mode.value).rglob(SUBJECT_OUTPUT_PER_RANK_PREFIX + "*"))
        result_file = outputs_folder / mode.value / SUBJECT_METRICS_FILE_NAME
        with result_file.open("a") as f:
            for i, file in enumerate(temp_files):
                temp_file_contents = file.read_text()
                if i == 0:
                    # Copy the first file as-is, including the first line with the column headers
                    f.write(temp_file_contents)
                else:
                    # For all files but the first one, cut off the header line.
                    f.write("\n".join(temp_file_contents.splitlines()[1:]))

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeVar

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.common_util import SUBJECT_METRICS_FILE_NAME, change_working_directory
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML.common import ModelExecutionMode, RECOVERY_CHECKPOINT_FILE_NAME, cleanup_checkpoint_folder
from InnerEye.ML.deep_learning_config import ARGS_TXT, VISUALIZATION_FOLDER
from InnerEye.ML.lightning_base import InnerEyeContainer, InnerEyeLightning
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.lightning_loggers import AzureMLLogger, StoringLogger
from InnerEye.ML.lightning_models import SUBJECT_OUTPUT_PER_RANK_PREFIX, ScalarLightning, \
    get_subject_output_file_per_rank
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler

TEMP_PREFIX = "temp/"

T = TypeVar('T')


def is_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP rank zero, before the training has actually started,
    by looking at environment variables.
    :return: True if the current process is global_rank 0.
    """
    global_rank = os.getenv("GLOBAL_RANK")
    local_rank = os.getenv("LOCAL_RANK")
    # When doing multi-node training, this indicates which node the present job is on. This is set in
    # set_environment_variables_for_multi_node
    node_rank = os.getenv("NODE_RANK", "0")
    return global_rank is None and local_rank is None and node_rank == "0"


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
                             **kwargs: Dict[str, Any]) -> \
        Tuple[Trainer, Optional[StoringLogger]]:
    """
    Creates a Pytorch Lightning Trainer object for the given model configuration. It creates checkpoint handlers
    and loggers. That includes a diagnostic logger for use in unit tests, that is also returned as the second
    return value.
    :param container: The container with model and data.
    :param resume_from_checkpoint: If provided, training resumes from this checkpoint point.
    :param num_nodes: The number of nodes to use in distributed training.
    :param kwargs: Any additional keyowrd arguments will be passed to the constructor of Trainer.
    :return: A tuple [Trainer object, diagnostic logger]
    """
    # For now, stick with the legacy behaviour of always saving only the last epoch checkpoint. For large segmentation
    # models, this still appears to be the best way of choosing them because validation loss on the relatively small
    # training patches is not stable enough. Going by the validation loss somehow works for the Prostate model, but
    # not for the HeadAndNeck model.
    best_checkpoint_callback = ModelCheckpoint(dirpath=str(container.checkpoint_folder),
                                               # filename=BEST_CHECKPOINT_FILE_NAME,
                                               # monitor=f"{VALIDATION_PREFIX}{MetricType.LOSS.value}",
                                               # save_top_k=1,
                                               save_last=True)
    # Recovery checkpoints: {epoch} will turn into a string like "epoch=1"
    # Store 1 recovery checkpoint every recovery_checkpoint_save_interval epochs. Due to a bug in Lightning, this
    # will still write alternate files recovery.ckpt and recovery-v0.ckpt, which are cleaned up later in
    # cleanup_checkpoint_folder
    recovery_checkpoint_callback = ModelCheckpoint(dirpath=str(container.checkpoint_folder),
                                                   filename=RECOVERY_CHECKPOINT_FILE_NAME,
                                                   period=container.recovery_checkpoint_save_interval
                                                   )

    num_gpus = torch.cuda.device_count() if container.use_gpu else 0
    logging.info(f"Number of available GPUs: {num_gpus}")
    if 0 <= container.max_num_gpus < num_gpus:
        num_gpus = container.max_num_gpus
        logging.info(f"Restricting the number of GPUs to {num_gpus}")
    # Accelerator should be "ddp" when running large models in AzureML (when using DDP_spawn, we get out of GPU memory).
    # For unit tests, only "ddp_spawn" works
    accelerator = "ddp" if num_gpus * num_nodes > 1 else None
    logging.info(f"Using {num_gpus} GPUs with accelerator '{accelerator}'")
    tensorboard_logger = TensorBoardLogger(save_dir=str(container.logs_folder), name="Lightning", version="")
    loggers = [tensorboard_logger, AzureMLLogger()]
    storing_logger: Optional[StoringLogger]
    if isinstance(container, InnerEyeContainer):
        storing_logger = StoringLogger()
        loggers.append(storing_logger)
    else:
        storing_logger = None
    # Use 32bit precision when running on CPU. Otherwise, make it depend on use_mixed_precision flag.
    precision = 32 if num_gpus == 0 else 16 if container.use_mixed_precision else 32
    # The next two flags control the settings in torch.backends.cudnn.deterministic and torch.backends.cudnn.benchmark
    # https://pytorch.org/docs/stable/notes/randomness.html
    # For the classification models, we observed only a small performance deterioration (increase in 10sec on total
    # training time of 22min) when switching to deterministic.
    if container.pl_deterministic:
        deterministic = True
        benchmark = False
    else:
        deterministic = False
        benchmark = True
    # Read out additional model-specific args here.
    # We probably want to keep essential ones like numgpu and logging.
    trainer = Trainer(default_root_dir=str(container.outputs_folder),
                      deterministic=deterministic,
                      benchmark=benchmark,
                      accelerator=accelerator,
                      max_epochs=container.num_epochs,
                      num_sanity_val_steps=container.pl_num_sanity_val_steps,
                      callbacks=[best_checkpoint_callback, recovery_checkpoint_callback],
                      logger=loggers,
                      progress_bar_refresh_rate=container.pl_progress_bar_refresh_rate,
                      num_nodes=num_nodes,
                      gpus=num_gpus,
                      precision=precision,
                      sync_batchnorm=True,
                      terminate_on_nan=container.detect_anomaly,
                      resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint else None,
                      **kwargs)
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


def model_train(checkpoint_handler: CheckpointHandler,
                container: LightningContainer,
                num_nodes: int = 1) -> Tuple[Trainer, Optional[StoringLogger]]:
    """
    The main training loop. It creates the Pytorch model based on the configuration options passed in,
    creates a Pytorch Lightning trainer, and trains the model.
    If a checkpoint was specified, then it loads the checkpoint before resuming training.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    :param num_nodes: The number of nodes to use in distributed training.
    :param container: A container object that holds the training data in PyTorch Lightning format
    and the model to train.
    :return: A tuple of [Trainer, StoringLogger]. Trainer is the Lightning Trainer object that was used for fitting
    the model. The StoringLogger object is returned when training an InnerEye built-in model, this is None when
    fitting other models.
    """
    # Get the path to the checkpoint to recover from
    checkpoint_path = checkpoint_handler.get_recovery_path_train()
    lightning_model = container.model

    container.before_training_on_all_ranks()
    resource_monitor: Optional[ResourceMonitor] = None
    # Execute some bookkeeping tasks only once if running distributed:
    if is_rank_zero():
        logging.info(f"Model checkpoints are saved at {container.checkpoint_folder}")
        container.before_training_on_rank_zero()
        write_args_file(container.config if isinstance(container, InnerEyeContainer) else container,
                        outputs_folder=container.outputs_folder)
        if container.monitoring_interval_seconds > 0:
            resource_monitor = start_resource_monitor(container)

    # Create the trainer object. Backup the environment variables before doing that, in case we need to run a second
    # training in the unit tests.d
    old_environ = dict(os.environ)
    # Set random seeds just before training. For segmentation models, we have
    # something that changes the random seed in the before_training_on_rank_zero hook.
    seed_everything(container.get_effective_random_seed())
    trainer, storing_logger = create_lightning_trainer(container,
                                                       checkpoint_path,
                                                       num_nodes=num_nodes,
                                                       **container.get_trainer_arguments())
    logging.info(f"GLOBAL_RANK: {os.getenv('GLOBAL_RANK')}, LOCAL_RANK {os.getenv('LOCAL_RANK')}. "
                 f"trainer.global_rank: {trainer.global_rank}")
    # InnerEye models use this logger for diagnostics
    if isinstance(lightning_model, InnerEyeLightning):
        if storing_logger is None:
            raise ValueError("InnerEye models require the storing_logger for diagnostics")
        lightning_model.storing_logger = storing_logger

    logging.info("Starting training")
    # When training models that are not built-in InnerEye models, we have no guarantee that they write
    # files to the right folder. Best guess is to change the current working directory to where files should go.
    data_module = container.get_data_module()
    with change_working_directory(container.outputs_folder):
        trainer.fit(lightning_model, datamodule=data_module)
        trainer.logger.close()  # type: ignore
    world_size = getattr(trainer, "world_size", 0)
    is_azureml_run = not is_offline_run_context(RUN_CONTEXT)
    # Per-subject model outputs for regression models are written per rank, and need to be aggregated here.
    # Each thread per rank will come here, and upload its files to the run outputs. Rank 0 will later download them.
    if is_azureml_run and world_size > 1 and isinstance(lightning_model, ScalarLightning):
        upload_output_file_as_temp(lightning_model.train_subject_outputs_logger.csv_path, container.outputs_folder)
        upload_output_file_as_temp(lightning_model.val_subject_outputs_logger.csv_path, container.outputs_folder)
    # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
    # We can now use the global_rank of the Lightining model, rather than environment variables, because DDP has set
    # all necessary properties.
    if lightning_model.global_rank != 0:
        logging.info(f"Terminating training thread with rank {lightning_model.global_rank}.")
        sys.exit()

    logging.info("Choosing the best checkpoint and removing redundant files.")
    cleanup_checkpoint_folder(container.checkpoint_folder)
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

    # Since we have trained the model further, let the checkpoint_handler object know so it can handle
    # checkpoints correctly.
    checkpoint_handler.additional_training_done()

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
    one single metrics file. Results is saved in config.outputs_folder / mode.value / SUBJECT_METRICS_FILE_NAME.
    This is done for the metrics files for training and for validation data separately.
    :param config: model config
    """
    for mode in [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL]:
        temp_files = (outputs_folder / mode.value).rglob(SUBJECT_OUTPUT_PER_RANK_PREFIX + "*")
        result_file = outputs_folder / mode.value / SUBJECT_METRICS_FILE_NAME
        result_file = result_file.open("a")
        for i, file in enumerate(temp_files):
            temp_file_contents = file.read_text()
            if i == 0:
                # Copy the first file as-is, including the first line with the column headers
                result_file.write(temp_file_contents)
            else:
                # For all files but the first one, cut off the header line.
                result_file.write(os.linesep + os.linesep.join(temp_file_contents.splitlines()[1:]))
        result_file.close()

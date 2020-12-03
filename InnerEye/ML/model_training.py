#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
from typing import Tuple, TypeVar

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.mlflow import MLFlowLogger

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.common_util import logging_section
from InnerEye.Common.metrics_dict import MetricType
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML.deep_learning_config import VISUALIZATION_FOLDER
from InnerEye.ML.lightning_models import StoringLogger, TRAIN_PREFIX, TrainingAndValidationDataLightning, \
    VALIDATION_PREFIX, create_lightning_model
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.utils.model_util import generate_and_print_model_summary
from InnerEye.ML.utils.training_util import ModelOutputsAndMetricsForEpoch, ModelTrainingResults
from InnerEye.ML.visualizers.patch_sampling import visualize_random_crops_for_dataset

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

T = TypeVar('T')


def is_rank_zero() -> bool:
    """
    Tries to guess if the current process is running as DDP rank zero, before the training has actually started,
    by looking at environment variables.
    :return: True if the current process is global_rank 0.
    """
    global_rank = os.getenv('GLOBAL_RANK')
    local_rank = os.getenv('LOCAL_RANK')
    return global_rank is None and local_rank is None


def model_train(config: ModelConfigBase,
                checkpoint_handler: CheckpointHandler) -> ModelTrainingResults:
    """
    The main training loop. It creates the Pytorch model based on the configuration options passed in,
    creates a Pytorch Lightning trainer, and trains the model.
    If a checkpoint was specified, then it loads the checkpoint before resuming training.
    :param config: The arguments which specify all required information.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    """
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.checkpoint_folder),
        filename='best_val_loss_checkpoint',
        monitor=f"{VALIDATION_PREFIX}{MetricType.LOSS.value}",
        save_last=True)
    num_gpus = torch.cuda.device_count() if config.use_gpu else 0
    accelerator = "ddp" if num_gpus > 1 else None
    logging.info(f"Using {num_gpus} GPUs with accelerator '{accelerator}'")
    storing_logger = StoringLogger()
    loggers = [storing_logger,
               TensorBoardLogger(save_dir=str(config.logs_folder), name="Lightning", version="")]
    if not is_offline_run_context(RUN_CONTEXT):
        mlflow_logger = MLFlowLogger(experiment_name=RUN_CONTEXT.experiment.name,
                                     tracking_uri=RUN_CONTEXT.experiment.workspace.get_mlflow_tracking_uri())
        # The MLFlow logger needs to get its ID from the AzureML run context, otherwise there will be two sets of
        # results for each run, one from native AzureML and one from the MLFlow logger.
        mlflow_logger._run_id = RUN_CONTEXT.id
        loggers.append(mlflow_logger)

    trainer = Trainer(default_root_dir=str(config.outputs_folder),
                      accelerator=accelerator,
                      max_epochs=config.num_epochs,
                      num_sanity_val_steps=0,  # Otherwise a small number of validation steps is run before first train
                      callbacks=[checkpoint_callback],
                      logger=loggers,
                      progress_bar_refresh_rate=0,  # Disable the progress bar,
                      # TODO antonsc: review. Some tests fail without this option
                      gpus=num_gpus,
                      terminate_on_nan=config.detect_anomaly,
                      )

    logging.info(f"GLOBAL_RANK: {os.getenv('GLOBAL_RANK')}, LOCAL_RANK {os.getenv('LOCAL_RANK')}. "
                 f"trainer.global_rank: {trainer.global_rank}")
    ml_util.set_random_seed(config.get_effective_random_seed(), "Model training")
    logging.debug("Creating the PyTorch model.")
    lightning_model = create_lightning_model(config)
    lightning_model.storing_logger = storing_logger

    resource_monitor = None
    # Execute some bookkeeping tasks only once if running distributed:
    if is_rank_zero():
        config.write_args_file()
        logging.info(str(config))
        # Save the dataset files for later use in cross validation analysis
        config.write_dataset_files()
        logging.info(f"Model checkpoints are saved at {config.checkpoint_folder}")

        # set the random seed for all libraries
        ml_util.set_random_seed(config.get_effective_random_seed(), "Patch visualization")
        # Visualize how patches are sampled for segmentation models. This changes the random generator, but we don't
        # want training to depend on how many patients we visualized, and hence set the random seed again right after.
        with logging_section("Visualizing the effect of sampling random crops for training"):
            visualize_random_crops_for_dataset(config)

        # Print out a detailed breakdown of layers, memory consumption and time.
        generate_and_print_model_summary(config, lightning_model.model)

        if config.monitoring_interval_seconds > 0:
            # initialize and start GPU monitoring
            diagnostics_events = config.logs_folder / "diagnostics"
            logging.info(f"Starting resource monitor, outputting to {diagnostics_events}")
            resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                               tensorboard_folder=diagnostics_events)
            resource_monitor.start()

    # TODO antonsc: Enable initializing the trainer from a checkpoint
    checkpoint_path = checkpoint_handler.get_recovery_path_train()

    # Training loop
    logging.info("Starting training")

    lightning_data = TrainingAndValidationDataLightning(config)
    # TODO: Why can't we do that in the constructor?
    lightning_data.config = config
    trainer.fit(lightning_model,
                datamodule=lightning_data,
                )
    # DDP will start multiple instances of the runner, one for each GPU. Those should terminate here after training.
    # We can now use the global_rank of the Lightining model, rather than environment variables, because DDP has set
    # all necessary properties.
    if lightning_model.global_rank != 0:
        logging.info(f"Terminating training thread with rank {lightning_model.global_rank}.")
        sys.exit()
    model_training_results = ModelTrainingResults(
        train_results_per_epoch=list(storing_logger.to_metrics_dicts(prefix_filter=TRAIN_PREFIX).values()),
        val_results_per_epoch=list(storing_logger.to_metrics_dicts(prefix_filter=VALIDATION_PREFIX).values()),
        train_diagnostics=lightning_model.train_diagnostics,
        val_diagnostics=lightning_model.val_diagnostics,
        optimal_temperature_scale_values_per_checkpoint_epoch=[]
    )

    logging.info("Finished training")

    # Since we have trained the model further, let the checkpoint_handler object know so it can handle
    # checkpoints correctly.
    checkpoint_handler.additional_training_done()

    # Upload visualization directory to AML run context to be able to see it
    # in the Azure UI.
    if config.max_batch_grad_cam > 0 and config.visualization_folder.exists():
        RUN_CONTEXT.upload_folder(name=VISUALIZATION_FOLDER, path=str(config.visualization_folder))

    lightning_model.close_all_loggers()
    if resource_monitor:
        # stop the resource monitoring process
        logging.info("Shutting down the resource monitor process. Aggregate resource utilization:")
        for name, value in resource_monitor.read_aggregate_metrics():
            logging.info(f"{name}: {value}")
            if not is_offline_run_context(RUN_CONTEXT):
                RUN_CONTEXT.log(name, value)
        resource_monitor.kill()

    return model_training_results


def temperature_scaling_steps(val_results_for_epoch: ModelOutputsAndMetricsForEpoch) -> \
        Tuple[float, ModelOutputsAndMetricsForEpoch]:
    """
    Perform the steps required for temperature scaling:
    1) Learn the temperature parameter on the logits and labels of the provided validation epoch
    2) Re-run the validation after learning the scaling parameter.

    :param config: Config for a sequence model.
    :param train_val_params: Train/Validate parameters to use.
    :param val_results_for_epoch: results from the validation epoch to use in order to perform temperature scaling.
    :return: the optimal temperature value and the validation results after scaling has been performed.
    """
    # re-create the training steps for the repeat pass, but with metrics saving enabled
    training_steps = None  # create_model_training_steps(config, train_val_params)
    # make sure results for a validation epoch have been passed in
    assert val_results_for_epoch.is_train is False
    # perform temperature scaling
    logits = val_results_for_epoch.get_logits()
    labels = val_results_for_epoch.get_labels()
    temperature_value = training_steps.learn_temperature_scale_parameter(logits, labels)
    # recompute the validation set results for the temperature scaled model
    val_epoch_results = None  # Should evaluate on validation set

    return temperature_value, val_epoch_results

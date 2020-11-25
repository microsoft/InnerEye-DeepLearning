#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Tuple, TypeVar, Union

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from InnerEye.Azure.azure_util import RUN_CONTEXT, is_offline_run_context
from InnerEye.Common.common_util import logging_section
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML.deep_learning_config import VISUALIZATION_FOLDER
from InnerEye.ML.lightning_models import InnerEyeLightning, TrainingAndValidationDataLightning, create_lightning_model
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training_steps import ModelTrainingStepsForSequenceModel
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.utils.model_util import generate_and_print_model_summary
from InnerEye.ML.utils.training_util import ModelOutputsAndMetricsForEpoch, ModelTrainingResults
from InnerEye.ML.visualizers.patch_sampling import visualize_random_crops_for_dataset

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

T = TypeVar('T')


def model_train(config: ModelConfigBase,
                checkpoint_handler: CheckpointHandler) -> ModelTrainingResults:
    """
    The main training loop. It creates the Pytorch model based on the configuration options passed in,
    creates a Pytorch Lightning trainer, and trains the model.
    If a checkpoint was specified, then it loads the checkpoint before resuming training.
    :param config: The arguments which specify all required information.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    """
    # Save the dataset files for later use in cross validation analysis
    config.write_dataset_files()

    # set the random seed for all libraries
    ml_util.set_random_seed(config.get_effective_random_seed(), "Patch visualization")
    # Visualize how patches are sampled for segmentation models. This changes the random generator, but we don't
    # want training to depend on how many patients we visualized, and hence set the random seed again right after.
    with logging_section("Visualizing the effect of sampling random crops for training"):
        visualize_random_crops_for_dataset(config)
    ml_util.set_random_seed(config.get_effective_random_seed(), "Model training")

    logging.debug("Creating the PyTorch model.")
    lightning_model = create_lightning_model(config)

    # Print out a detailed breakdown of layers, memory consumption and time.
    assert isinstance(lightning_model, InnerEyeLightning)
    generate_and_print_model_summary(config, lightning_model.model)

    logging.info(f"Model checkpoints are saved at {config.checkpoint_folder}")
    config.create_loggers_for_training()

    # Get the path to the checkpoint to recover from
    checkpoint_path = checkpoint_handler.get_recovery_path_train()

    # models_and_optimizer = ModelAndInfo(config=config,
    #                                     model_execution_mode=ModelExecutionMode.TRAIN,
    #                                     checkpoint_path=checkpoint_path)
    #
    # # Create the main model
    # # If continuing from a previous run at a specific epoch, then load the previous model.
    # model_loaded = models_and_optimizer.try_create_model_and_load_from_checkpoint()
    # if not model_loaded:
    #     raise ValueError("There was no checkpoint file available for the model for given start_epoch {}"
    #                      .format(config.start_epoch))

    # # Create the mean teacher model and move to GPU
    # if config.compute_mean_teacher_model:
    #     mean_teacher_model_loaded =
    #     models_and_optimizer.try_create_mean_teacher_model_load_from_checkpoint_and_adjust()
    #     if not mean_teacher_model_loaded:
    #         raise ValueError("There was no checkpoint file available for the mean teacher model "
    #                          f"for given start_epoch {config.start_epoch}")

    # Training loop
    logging.info("Starting training")

    resource_monitor = None
    if config.monitoring_interval_seconds > 0:
        # initialize and start GPU monitoring
        diagnostics_events = config.logs_folder / "diagnostics"
        logging.info(f"Starting resource monitor, outputting to {diagnostics_events}")
        resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                           tensorboard_folder=diagnostics_events)
        resource_monitor.start()

    optimal_temperature_scale_values = []

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(config.checkpoint_folder),
        filename='best_val_loss_checkpoint',
        monitor='val_loss',
        save_last=True)
    trainer = Trainer(default_root_dir=str(config.outputs_folder),
                      max_epochs=config.num_epochs,
                      num_sanity_val_steps=0,  # Otherwise a small number of validation steps is run before first train
                      logger=TensorBoardLogger(save_dir=str(config.logs_folder), name="Lightning", version=""),
                      callbacks=[checkpoint_callback],
                      progress_bar_refresh_rate=0,  # Disable the progress bar,
                      # TODO antonsc: review. Some tests fail without this option
                      gpus=0,
                      terminate_on_nan=config.detect_anomaly,
                      )
    lightning_data = TrainingAndValidationDataLightning(config)
    # TODO: Why can't we do that in the constructor?
    lightning_data.config = config
    trainer.fit(lightning_model,
                datamodule=lightning_data,
                )
    model_training_results = ModelTrainingResults(
        train_results_per_epoch=lightning_model.training_metrics_per_epoch,
        val_results_per_epoch=lightning_model.validation_metrics_per_epoch,
        optimal_temperature_scale_values_per_checkpoint_epoch=optimal_temperature_scale_values
    )

    logging.info("Finished training")

    # Since we have trained the model further, let the checkpoint_handler object know so it can handle
    # checkpoints correctly.
    checkpoint_handler.additional_training_done()

    # Upload visualization directory to AML run context to be able to see it
    # in the Azure UI.
    if config.max_batch_grad_cam > 0 and config.visualization_folder.exists():
        RUN_CONTEXT.upload_folder(name=VISUALIZATION_FOLDER, path=str(config.visualization_folder))

    config.close_all_loggers()
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
    assert isinstance(training_steps, ModelTrainingStepsForSequenceModel)
    # make sure results for a validation epoch have been passed in
    assert val_results_for_epoch.is_train is False
    # perform temperature scaling
    logits = val_results_for_epoch.get_logits()
    labels = val_results_for_epoch.get_labels()
    temperature_value = training_steps.learn_temperature_scale_parameter(logits, labels)
    # recompute the validation set results for the temperature scaled model
    val_epoch_results = None  # Should evaluate on validation set

    return temperature_value, val_epoch_results

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import os
from time import time
from typing import Optional, Tuple, TypeVar

from torch.cuda.amp import GradScaler

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.Common.common_util import empty_string_to_none
from InnerEye.Common.metrics_dict import MetricsDict
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML import metrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import VISUALIZATION_FOLDER
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training_steps import ModelTrainingStepsBase, \
    ModelTrainingStepsForScalarModel, ModelTrainingStepsForSegmentation, ModelTrainingStepsForSequenceModel, \
    TrainValidateParameters
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.metrics_util import create_summary_writers
from InnerEye.ML.utils.ml_util import RandomStateSnapshot
from InnerEye.ML.utils.model_util import ModelAndInfo, generate_and_print_model_summary, save_checkpoint
from InnerEye.ML.utils.run_recovery import RunRecovery, get_recovery_path_train
from InnerEye.ML.utils.training_util import ModelOutputsAndMetricsForEpoch, ModelTrainingResults

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

T = TypeVar('T')


def model_train(config: ModelConfigBase, run_recovery: Optional[RunRecovery] = None) -> ModelTrainingResults:
    """
    The main training loop. It creates the model, dataset, optimizer_type, and criterion, then proceeds
    to train the model. If a checkpoint was specified, then it loads the checkpoint before resuming training.

    :param config: The arguments which specify all required information.
    :param run_recovery: Recovery information to restart training from an existing run.
    :raises TypeError: If the arguments are of the wrong type.
    :raises ValueError: When there are issues loading a previous checkpoint.
    """
    # Save the dataset files for later use in cross validation analysis
    config.write_dataset_files()

    # set the random seed for all libraries
    ml_util.set_random_seed(config.get_effective_random_seed(), "Model Training")

    logging.debug("Creating the PyTorch model.")

    # Create the train loader and validation loader to load images from the dataset
    data_loaders = config.create_data_loaders()

    # Create models, optimizers, and whether is_mean_teacher
    checkpoint_path = get_recovery_path_train(run_recovery=run_recovery,
                                              is_mean_teacher=False,
                                              epoch=config.start_epoch)
    models_and_optimizers = [ModelAndInfo(config=config,
                                          model_execution_mode=ModelExecutionMode.TRAIN,
                                          is_mean_teacher=False,
                                          checkpoint_path=checkpoint_path if config.should_load_checkpoint_for_training() else None)]

    if config.compute_mean_teacher_model:
        checkpoint_path = get_recovery_path_train(run_recovery=run_recovery,
                                                  is_mean_teacher=True,
                                                  epoch=config.start_epoch)
        models_and_optimizers.append(ModelAndInfo(config=config,
                                                  model_execution_mode=ModelExecutionMode.TRAIN,
                                                  is_mean_teacher=True,
                                                  checkpoint_path=checkpoint_path if config.should_load_checkpoint_for_training() else None))

    # Create the models.
    # If continuing from a previous run at a specific epoch, then load the previous model.
    for model_and_info in models_and_optimizers:
        model_loaded = model_and_info.try_create_model_and_load_from_checkpoint()
        if not model_loaded:
            raise ValueError("There was no checkpoint file available for the model for given start_epoch {}"
                             .format(config.start_epoch))

    # Print out a detailed breakdown of layers, memory consumption and time.
    generate_and_print_model_summary(config, models_and_optimizers[0].model)

    # Move model to GPU and adjust for multiple GPUs
    models_and_optimizers[0].adjust_model_for_gpus()
    if len(models_and_optimizers) > 1:
        models_and_optimizers[1].create_summary_and_adjust_model_for_gpus()

    # Create optimizer
    optimizer_loaded = models_and_optimizers[0].try_create_optimizer_and_load_from_checkpoint()
    if not optimizer_loaded:
        raise ValueError("There was no checkpoint file available for the optimizer for given start_epoch {}"
                         .format(config.start_epoch))

    # Create checkpoint directory for this run if it doesn't already exist
    logging.info("Models are saved at {}".format(config.checkpoint_folder))
    if not os.path.isdir(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder)

    # Create the SummaryWriters for Tensorboard
    writers = create_summary_writers(config)
    config.create_dataframe_loggers()

    model = models_and_optimizers[0].model
    optimizer = models_and_optimizers[0].optimizer
    mean_teacher_model = models_and_optimizers[1].model if len(models_and_optimizers) > 1 else None

    # Create LR scheduler
    l_rate_scheduler = SchedulerWithWarmUp(config, optimizer)

    # Training loop
    logging.info("Starting training")
    train_results_per_epoch, val_results_per_epoch, learning_rates_per_epoch = [], [], []

    resource_monitor = None
    if config.monitoring_interval_seconds > 0:
        # initialize and start GPU monitoring
        resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                           tb_log_file_path=str(config.logs_folder / "diagnostics"))
        resource_monitor.start()

    gradient_scaler = GradScaler() if config.use_gpu and config.use_mixed_precision else None
    optimal_temperature_scale_values = []
    for epoch in config.get_train_epochs():
        logging.info("Starting epoch {}".format(epoch))
        save_epoch = config.should_save_epoch(epoch) and optimizer is not None

        # store the learning rates used for each epoch
        epoch_lrs = l_rate_scheduler.get_last_lr()
        learning_rates_per_epoch.append(epoch_lrs)

        train_val_params: TrainValidateParameters = \
            TrainValidateParameters(data_loader=data_loaders[ModelExecutionMode.TRAIN],
                                    model=model,
                                    mean_teacher_model=mean_teacher_model,
                                    epoch=epoch,
                                    optimizer=optimizer,
                                    gradient_scaler=gradient_scaler,
                                    epoch_learning_rate=epoch_lrs,
                                    summary_writers=writers,
                                    dataframe_loggers=config.metrics_data_frame_loggers,
                                    in_training_mode=True)
        training_steps = create_model_training_steps(config, train_val_params)
        train_epoch_results = train_or_validate_epoch(training_steps)
        train_results_per_epoch.append(train_epoch_results.metrics)

        metrics.validate_and_store_model_parameters(writers.train, epoch, model)
        # Run without adjusting weights on the validation set
        train_val_params.in_training_mode = False
        train_val_params.data_loader = data_loaders[ModelExecutionMode.VAL]
        # if temperature scaling is enabled then do not save validation metrics for the checkpoint epochs
        # as these will be re-computed after performing temperature scaling on the validation set.
        if isinstance(config, SequenceModelBase):
            train_val_params.save_metrics = not (save_epoch and config.temperature_scaling_config)

        training_steps = create_model_training_steps(config, train_val_params)
        val_epoch_results = train_or_validate_epoch(training_steps)
        val_results_per_epoch.append(val_epoch_results.metrics)

        if config.is_segmentation_model:
            metrics.store_epoch_stats_for_segmentation(config.outputs_folder, epoch, epoch_lrs,
                                                       train_epoch_results.metrics,
                                                       val_epoch_results.metrics)

        if save_epoch:
            # perform temperature scaling if required
            if isinstance(config, SequenceModelBase) and config.temperature_scaling_config:
                optimal_temperature, scaled_val_results = \
                    temperature_scaling_steps(config, train_val_params, val_epoch_results)
                optimal_temperature_scale_values.append(optimal_temperature)
                # overwrite the metrics for the epoch with the metrics from the temperature scaled model
                val_results_per_epoch[-1] = scaled_val_results.metrics

            assert optimizer is not None
            save_checkpoint(model, optimizer, epoch, config)
            if config.compute_mean_teacher_model:
                assert mean_teacher_model is not None
                save_checkpoint(mean_teacher_model, optimizer, epoch, config, mean_teacher_model=True)

        # Updating the learning rate should happen at the end of the training loop, so that the
        # initial learning rate will be used for the very first epoch.
        l_rate_scheduler.step()

    model_training_results = ModelTrainingResults(
        train_results_per_epoch=train_results_per_epoch,
        val_results_per_epoch=val_results_per_epoch,
        learning_rates_per_epoch=learning_rates_per_epoch,
        optimal_temperature_scale_values_per_checkpoint_epoch=optimal_temperature_scale_values
    )

    logging.info("Finished training")

    # Upload visualization directory to AML run context to be able to see it
    # in the Azure UI.
    if config.max_batch_grad_cam > 0 and config.visualization_folder.exists():
        RUN_CONTEXT.upload_folder(name=VISUALIZATION_FOLDER, path=str(config.visualization_folder))

    writers.close_all()
    config.metrics_data_frame_loggers.close_all()
    if resource_monitor:
        # stop the resource monitoring process
        resource_monitor.kill()

    return model_training_results


def temperature_scaling_steps(config: SequenceModelBase,
                              train_val_params: TrainValidateParameters,
                              val_results_for_epoch: ModelOutputsAndMetricsForEpoch) -> \
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
    train_val_params.save_metrics = True
    training_steps = create_model_training_steps(config, train_val_params)
    assert isinstance(training_steps, ModelTrainingStepsForSequenceModel)
    # make sure results for a validation epoch have been passed in
    assert val_results_for_epoch.is_train is False
    # perform temperature scaling
    logits = val_results_for_epoch.get_logits()
    labels = val_results_for_epoch.get_labels()
    temperature_value = training_steps.learn_temperature_scale_parameter(logits, labels)
    # recompute the validation set results for the temperature scaled model
    val_epoch_results = train_or_validate_epoch(training_steps)

    return temperature_value, val_epoch_results


def train_or_validate_epoch(training_steps: ModelTrainingStepsBase) -> ModelOutputsAndMetricsForEpoch:
    """
    Trains or validates the model for one epoch.
    :param training_steps: Training pipeline to use.
    :returns: The results for training or validation. Result type depends on the type of model that is trained.
    """
    epoch_start_time = time()
    training_random_state = None
    train_val_params = training_steps.train_val_params
    config = training_steps.model_config
    if not train_val_params.in_training_mode:
        # take the snapshot of the existing random state
        training_random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation
        ml_util.set_random_seed(config.get_effective_random_seed(), "Model Training")

    status_string = "training" if train_val_params.in_training_mode else "validation"
    item_start_time = time()
    num_load_time_warnings = 0
    num_load_time_exceeded = 0
    num_batches = 0
    total_extra_load_time = 0.0
    total_load_time = 0.0
    model_outputs_epoch = []
    for batch_index, sample in enumerate(train_val_params.data_loader):
        item_finish_time = time()
        item_load_time = item_finish_time - item_start_time
        # Having slow minibatch loading is OK in the very first batch of the every epoch, where processes
        # are spawned. Later, the load time should be zero.
        if batch_index == 0:
            logging.info(f"Loaded the first minibatch of {status_string} data in {item_load_time:0.2f} sec.")
        elif item_load_time > MAX_ITEM_LOAD_TIME_SEC:
            num_load_time_exceeded += 1
            total_extra_load_time += item_load_time
            if num_load_time_warnings < MAX_LOAD_TIME_WARNINGS:
                logging.warning(f"Loading {status_string} minibatch {batch_index} took {item_load_time:0.2f} sec. "
                                f"This can mean that there are not enough data loader worker processes, or that there "
                                f"is a "
                                f"performance problem in loading. This warning will be printed at most "
                                f"{MAX_LOAD_TIME_WARNINGS} times.")
                num_load_time_warnings += 1
        model_outputs_minibatch = training_steps.forward_and_backward_minibatch(
            sample, batch_index, train_val_params.epoch)
        model_outputs_epoch.append(model_outputs_minibatch)
        train_finish_time = time()
        logging.debug(f"Epoch {train_val_params.epoch} {status_string} batch {batch_index}: "
                      f"Loaded in {item_load_time:0.2f}sec, "
                      f"{status_string} in {(train_finish_time - item_finish_time):0.2f}sec. "
                      f"Loss = {model_outputs_minibatch.loss}")
        total_load_time += item_finish_time - item_start_time
        num_batches += 1
        item_start_time = time()

    # restore the training random state when validation has finished
    if training_random_state is not None:
        training_random_state.restore_random_state()

    epoch_time_seconds = time() - epoch_start_time
    logging.info(f"Epoch {train_val_params.epoch} {status_string} took {epoch_time_seconds:0.2f} sec "
                 f"of which data loading took {total_load_time:0.2f} sec")
    if num_load_time_exceeded > 0:
        logging.warning("The dataloaders were not fast enough to always supply the next batch in less than "
                        f"{MAX_ITEM_LOAD_TIME_SEC}sec.")
        logging.warning(f"In this epoch, {num_load_time_exceeded} out of {num_batches} batches exceeded the load time "
                        f"threshold. The total loading time for the slow batches was {total_extra_load_time:0.2f}sec.")

    _metrics = training_steps.get_epoch_results_and_store(epoch_time_seconds) \
        if train_val_params.save_metrics else MetricsDict()
    return ModelOutputsAndMetricsForEpoch(
        metrics=_metrics,
        model_outputs=model_outputs_epoch,
        is_train=train_val_params.in_training_mode
    )


def create_model_training_steps(model_config: ModelConfigBase,
                                train_val_params: TrainValidateParameters) -> ModelTrainingStepsBase:
    """
    Create model training steps based on the model config and train/val parameters
    :param model_config: Model configs to use
    :param train_val_params: Train/Val parameters to use
    :return:
    """
    if isinstance(model_config, SegmentationModelBase):
        return ModelTrainingStepsForSegmentation(model_config, train_val_params)
    elif isinstance(model_config, ScalarModelBase):
        if isinstance(model_config, SequenceModelBase):
            return ModelTrainingStepsForSequenceModel(model_config, train_val_params)
        else:
            return ModelTrainingStepsForScalarModel(model_config, train_val_params)
    else:
        raise NotImplementedError(f"There are no model training steps defined for config type {type(model_config)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="The name of the model to train", type=empty_string_to_none,
                        required=True)
    args = parser.parse_args()
    model_train(ModelConfigLoader().create_model_config_from_name(args.model))


if __name__ == '__main__':
    main()

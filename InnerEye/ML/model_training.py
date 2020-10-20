#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import os
from time import time
from typing import Optional, Tuple, TypeVar

import torch

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
from InnerEye.ML.utils import ml_util, model_util
from InnerEye.ML.utils.aml_distributed_utils import get_global_rank, get_global_size, get_local_size, get_local_rank

from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.lr_scheduler import SchedulerWithWarmUp
from InnerEye.ML.utils.metrics_util import create_summary_writers
from InnerEye.ML.utils.ml_util import RandomStateSnapshot
from InnerEye.ML.utils.model_util import ModelAndInfo, generate_and_print_model_summary
from InnerEye.ML.utils.run_recovery import RunRecovery, get_recovery_path_train
from InnerEye.ML.utils.training_util import ModelOutputsAndMetricsForEpoch, ModelTrainingResults

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

T = TypeVar('T')


def load_checkpoint_from_model_and_info(run_recovery: Optional[RunRecovery], config: ModelConfigBase,
                                        model_and_info: ModelAndInfo) -> ModelAndInfo:
    is_mean_teacher = model_and_info.is_mean_teacher
    checkpoint_path = run_recovery.get_checkpoint_paths(config.start_epoch, is_mean_teacher)[0] \
        if run_recovery else config.get_path_to_checkpoint(config.start_epoch, is_mean_teacher)
    result = model_util.load_from_checkpoint_and_adjust(config, checkpoint_path, model_and_info)
    if result.checkpoint_epoch is None:
        raise ValueError("There was no checkpoint file available for the given start_epoch {}"
                         .format(config.start_epoch))
    return result


def model_train(config: ModelConfigBase,
                run_recovery: Optional[RunRecovery] = None) -> None:
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

    if config.use_ddp:

        world_size = get_global_size(config.is_offline_run)

        if config.is_offline_run:
            # set the environment variable for master node address
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            # spawn processes
            torch.multiprocessing.spawn(train,
                                        args=(config, run_recovery),
                                        nprocs=world_size)

        else:
            # AzureML MPI configuration handles rank
            train(None, config, run_recovery=run_recovery)
    else:
        single_process_rank = 0
        train(single_process_rank, config, run_recovery=run_recovery)


def train(rank: Optional[int], config: ModelConfigBase, run_recovery: Optional[RunRecovery] = None):
    """

    :param rank: The global rank of the current process (for DistributedDataParallel). For single process, rank=0
    :param model: The model to train.
    :param config: The arguments which specify all required information.
    :param run_recovery: Recovery information to restart training from an existing run.
    :return:
    """
    global_rank = get_global_rank() if rank is None else rank  # For 1 machine, this is same as local rank

    local_rank = global_rank if config.is_offline_run else get_local_rank()
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device('cpu')

    devices = config.get_cuda_devices()
    print(f'all visible cuda devices: {devices}')

    if config.use_ddp:
        world_size = get_global_size(config.is_offline_run)
        print(f"Running distributed training on device with global rank {global_rank} and local rank {local_rank}")
        torch.distributed.init_process_group(  # type: ignore
            backend=config.dist_backend,
            init_method=config.init_method,
            world_size=world_size,
            rank=global_rank)

        n_gpus_per_node = get_local_size()
        config.train_batch_size = int(config.train_batch_size // n_gpus_per_node)
        config.num_dataload_workers = int((config.num_dataload_workers + n_gpus_per_node - 1) / n_gpus_per_node)

        print(f'Updated batch size for mutiple GPUs: train_batch_size={config.train_batch_size},'
              f' num_dataload_workers={config.num_dataload_workers}')

    # Create the train loader and validation loader to load images from the dataset
    data_loaders = config.create_data_loaders()

    if config.use_ddp:
        train_dataset = data_loaders[ModelExecutionMode.TRAIN].dataset
        len_dataset = len(train_dataset)
        assert 2 * len_dataset >= world_size, f"2* len(dataset) (={2*len_dataset}) must be >= num GPUs (={world_size})"

    # Get the path to the checkpoint to recover from
    checkpoint_path = get_recovery_path_train(run_recovery=run_recovery,
                                              epoch=config.start_epoch)

    models_and_optimizer = ModelAndInfo(config=config,
                                        model_execution_mode=ModelExecutionMode.TRAIN,
                                        checkpoint_path=checkpoint_path if
                                        config.should_load_checkpoint_for_training() else None)

    # Create the main model
    # If continuing from a previous run at a specific epoch, then load the previous model.
    model_loaded = models_and_optimizer.try_create_model_and_load_from_checkpoint()
    if not model_loaded:
        raise ValueError("There was no checkpoint file available for the model for given start_epoch {}"
                         .format(config.start_epoch))

    # Print out a detailed breakdown of layers, memory consumption and time.
    generate_and_print_model_summary(config, models_and_optimizer.model, device)

    # Move model to GPU and adjust for multiple GPUs
    models_and_optimizer.adjust_model_for_gpus(rank=local_rank)

    # Create the mean teacher model and move to GPU
    if config.compute_mean_teacher_model:
        mean_teacher_model_loaded = models_and_optimizer.try_create_mean_teacher_model_load_from_checkpoint_and_adjust()
        if not mean_teacher_model_loaded:
            raise ValueError("There was no checkpoint file available for the mean teacher model for given start_epoch {}"
                             .format(config.start_epoch))

    # Create optimizer
    optimizer_loaded = models_and_optimizer.try_create_optimizer_and_load_from_checkpoint()
    if not optimizer_loaded:
        raise ValueError("There was no checkpoint file available for the optimizer for given start_epoch {}"
                         .format(config.start_epoch))

    # Create checkpoint directory for this run if it doesn't already exist
    logging.info("Models are saved at {}".format(config.checkpoint_folder))
    if not os.path.isdir(config.checkpoint_folder):
        os.makedirs(config.checkpoint_folder, exist_ok=True)

    # Create the SummaryWriters for Tensorboard
    writers = create_summary_writers(config, rank=global_rank)

    config.create_dataframe_loggers()

    # Create LR scheduler
    l_rate_scheduler = SchedulerWithWarmUp(config, models_and_optimizer.optimizer)

    # Training loop
    logging.info("Starting training")
    train_results_per_epoch, val_results_per_epoch, learning_rates_per_epoch = [], [], []

    resource_monitor = None
    if config.monitoring_interval_seconds > 0:
        # initialize and start GPU monitoring
        resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                           tb_log_file_path=str(config.logs_folder / "diagnostics"))
        resource_monitor.start()

    gradient_scaler = torch.cuda.amp.GradScaler() if config.use_gpu and config.use_mixed_precision else None
    optimal_temperature_scale_values = []
    for epoch in config.get_train_epochs():
        logging.info("Starting epoch {}".format(epoch))
        save_epoch = config.should_save_epoch(epoch) and models_and_optimizer.optimizer is not None

        if config.use_ddp:
            # set epoch for DistributedSampler to make shuffling work properly
            data_loaders[ModelExecutionMode.TRAIN].sampler.set_epoch(epoch)

        # store the learning rates used for each epoch
        epoch_lrs = l_rate_scheduler.get_last_lr()
        learning_rates_per_epoch.append(epoch_lrs)

        train_val_params: TrainValidateParameters = \
            TrainValidateParameters(data_loader=data_loaders[ModelExecutionMode.TRAIN],
                                    model=models_and_optimizer.model,
                                    mean_teacher_model=models_and_optimizer.mean_teacher_model,
                                    epoch=epoch,
                                    optimizer=models_and_optimizer.optimizer,
                                    gradient_scaler=gradient_scaler,
                                    epoch_learning_rate=epoch_lrs,
                                    summary_writers=writers,
                                    dataframe_loggers=config.metrics_data_frame_loggers,
                                    in_training_mode=True)



        training_steps = create_model_training_steps(config, train_val_params)
        train_epoch_results = train_or_validate_epoch(training_steps, local_rank, device)
        train_results_per_epoch.append(train_epoch_results.metrics)

        metrics.validate_and_store_model_parameters(writers.train, epoch, models_and_optimizer.model)
        # Run without adjusting weights on the validation set
        train_val_params.in_training_mode = False
        train_val_params.data_loader = data_loaders[ModelExecutionMode.VAL]
        # if temperature scaling is enabled then do not save validation metrics for the checkpoint epochs
        # as these will be re-computed after performing temperature scaling on the validation set.
        if isinstance(config, SequenceModelBase):
            train_val_params.save_metrics = not (save_epoch and config.temperature_scaling_config)

        training_steps = create_model_training_steps(config, train_val_params)
        val_epoch_results = train_or_validate_epoch(training_steps, local_rank, device)
        val_results_per_epoch.append(val_epoch_results.metrics)

        if config.is_segmentation_model:
            metrics.store_epoch_stats_for_segmentation(config.outputs_folder, epoch, epoch_lrs,
                                                       train_epoch_results.metrics,
                                                       val_epoch_results.metrics)

        if save_epoch and global_rank == 0:
            # perform temperature scaling if required
            if isinstance(config, SequenceModelBase) and config.temperature_scaling_config:
                optimal_temperature, scaled_val_results = \
                    temperature_scaling_steps(config, train_val_params, val_epoch_results)
                optimal_temperature_scale_values.append(optimal_temperature)
                # overwrite the metrics for the epoch with the metrics from the temperature scaled model
                val_results_per_epoch[-1] = scaled_val_results.metrics

            models_and_optimizer.save_checkpoint(epoch)

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

    # return model_training_results


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


def train_or_validate_epoch(training_steps: ModelTrainingStepsBase, rank, device) -> ModelOutputsAndMetricsForEpoch:
    """
    Trains or validates the model for one epoch.
    :param training_steps: Training pipeline to use.
    :returns: The results for training or validation. Result type depends on the type of model that is trained.
    """
    training_random_state = None
    train_val_params = training_steps.train_val_params
    config = training_steps.model_config
    cuda_available = torch.cuda.is_available() & rank == 0

    if cuda_available:
        item_start_time = torch.cuda.Event(enable_timing=True)
        item_finish_time = torch.cuda.Event(enable_timing=True)
        train_finish_time = torch.cuda.Event(enable_timing=True)
        epoch_start_time = torch.cuda.Event(enable_timing=True)
        epoch_end_time = torch.cuda.Event(enable_timing=True)
        epoch_start_time.record()
    else:
        epoch_start_time = time()

    if not train_val_params.in_training_mode:
        # take the snapshot of the existing random state
        training_random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation
        ml_util.set_random_seed(config.get_effective_random_seed(), "Model Training")

    status_string = "training" if train_val_params.in_training_mode else "validation"
    if cuda_available:
        item_start_time.record()
    else:
        item_start_time = time()

    num_load_time_warnings = 0
    num_load_time_exceeded = 0
    num_batches = 0
    total_extra_load_time = 0.0
    total_load_time = 0.0
    model_outputs_epoch = []
    for batch_index, sample in enumerate(train_val_params.data_loader):

        if cuda_available:
            item_finish_time.record()
            torch.cuda.synchronize()
            item_load_time = item_start_time.elapsed_time(item_finish_time)
        else:
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
            sample, batch_index, train_val_params.epoch, rank=rank, device=device)
        model_outputs_epoch.append(model_outputs_minibatch)
        if cuda_available:
            train_finish_time.record()
            torch.cuda.synchronize()
            status_time = item_finish_time.elapsed_time(train_finish_time)
        else:
            train_finish_time = time()
            status_time = train_finish_time - item_finish_time
        logging.debug(f"Epoch {train_val_params.epoch} {status_string} batch {batch_index}: "
                      f"Loaded in {item_load_time:0.2f}sec, "
                      f"{status_string} in {status_time:0.2f}sec. "
                      f"Loss = {model_outputs_minibatch.loss}")
        if cuda_available:
            torch.cuda.synchronize()
            total_load_time = item_start_time.elapsed_time(item_finish_time)
            item_start_time = torch.cuda.Event(enable_timing=True)
            item_start_time.record()
        else:
            total_load_time += item_finish_time - item_start_time
            item_start_time = time()
        num_batches += 1

    # restore the training random state when validation has finished
    if training_random_state is not None:
        training_random_state.restore_random_state()

    if cuda_available:
        epoch_end_time.record()
        torch.cuda.synchronize()
        epoch_time_seconds = epoch_start_time.elapsed_time(epoch_end_time)
    else:
        epoch_end_time = time()
        epoch_time_seconds = epoch_end_time - epoch_start_time

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

    model_config = ModelConfigLoader().create_model_config_from_name(args.model)

    model_train(model_config)


if __name__ == '__main__':
    main()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import os
from dataclasses import dataclass
from time import time
from typing import List, Optional, TypeVar

from InnerEye.Azure.azure_util import RUN_CONTEXT
from InnerEye.Common import common_util
from InnerEye.Common.common_util import empty_string_to_none
from InnerEye.Common.metrics_dict import MetricsDict
from InnerEye.Common.resource_monitor import ResourceMonitor
from InnerEye.ML import metrics
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import VISUALIZATION_FOLDER
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.model_training_steps import ModelTrainingStepsBase, ModelTrainingStepsForScalarModel, \
    ModelTrainingStepsForSegmentation, ModelTrainingStepsForSequenceModel, TrainValidateParameters
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import ml_util, model_util
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.lr_scheduler import LRScheduler
from InnerEye.ML.utils.metrics_util import create_summary_writers
from InnerEye.ML.utils.ml_util import RandomStateSnapshot
from InnerEye.ML.utils.model_util import ModelAndInfo, generate_and_print_model_summary, save_checkpoint
from InnerEye.ML.utils.run_recovery import RunRecovery

MAX_ITEM_LOAD_TIME_SEC = 0.5
MAX_LOAD_TIME_WARNINGS = 3

T = TypeVar('T')


@dataclass(frozen=True)
class ModelTrainingResult:
    """
    Stores the results from training, with the results on training and validation data for each training epoch.
    """
    train_results_per_epoch: List[MetricsDict]
    val_results_per_epoch: List[MetricsDict]
    learning_rates_per_epoch: List[List[float]]

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

        if len(self.train_results_per_epoch) != len(self.val_results_per_epoch) != len(self.learning_rates_per_epoch):
            raise Exception("train_results_per_epoch must be the same length as val_results_per_epoch found "
                            "and learning_rates_per_epoch, found: train_metrics_per_epoch={}, "
                            "val_metrics_per_epoch={}, learning_rates_per_epoch={}"
                            .format(len(self.train_results_per_epoch), len(self.val_results_per_epoch),
                                    len(self.learning_rates_per_epoch)))


def load_checkpoint(run_recovery: RunRecovery, config: ModelConfigBase,
                    model_and_info: ModelAndInfo) -> ModelAndInfo:
    is_mean_teacher = model_and_info.is_mean_teacher
    checkpoint_path = run_recovery.get_checkpoint_paths(config.start_epoch, is_mean_teacher)[0] \
        if run_recovery else config.get_path_to_checkpoint(config.start_epoch, is_mean_teacher)
    result = model_util.load_from_checkpoint_and_adjust(config, checkpoint_path, model_and_info)
    if result.checkpoint_epoch is None:
        raise ValueError("There was no checkpoint file available for the given start_epoch {}"
                         .format(config.start_epoch))
    return result


def model_train(config: ModelConfigBase, run_recovery: Optional[RunRecovery] = None) -> ModelTrainingResult:
    """
    The main training loop. It creates the model, dataset, optimizer_type, and criterion, then proceeds
    to train the model. If a checkpoint was specified, then it loads the checkpoint before resuming training.

    :param config: The arguments which specify all required information.
    :param run_recovery: Recovery information to restart training from an existing run.
    :raises TypeError: If the arguments are of the wrong type.
    :raises ValueError: When there are issues loading a previous checkpoint.
    """
    # save the datasets csv for record
    config.write_dataset_files()

    # set the random seed for all libraries
    ml_util.set_random_seed(config.random_seed)

    logging.debug("Creating the pytorch model.")

    # Create the train loader and validation loader to load images from the dataset
    data_loaders = config.create_data_loaders()

    # Create models, optimizers, and whether is_mean_teacher
    model = config.create_model()
    models_and_optimizers = [ModelAndInfo(model, model_util.create_optimizer(config, model))]
    if config.compute_mean_teacher_model:
        models_and_optimizers.append(ModelAndInfo(config.create_model(), is_mean_teacher=True))

    # If continuing from a previous run at a specific epoch, then load the previous model
    if config.should_load_checkpoint_for_training():
        assert run_recovery is not None
        models_and_optimizers = [load_checkpoint(run_recovery, config, model_and_info)
                                 for model_and_info in models_and_optimizers]
    # Otherwise, create checkpoint directory for this run
    else:
        logging.info("Models are saved at {}".format(config.checkpoint_folder))
        if not os.path.isdir(config.checkpoint_folder):
            os.makedirs(config.checkpoint_folder)

    # Print out a detailed breakdown of layers, memory consumption and time.
    generate_and_print_model_summary(config, model)

    # Enable mixed precision training and data parallelization (no-op if already done).
    # This relies on the information generated in the model summary.
    # We only want to do this if we didn't call load_checkpoint above, because attempting updating twic
    # causes an error.
    models_and_optimizers = [model_util.update_model_for_mixed_precision_and_parallel(model_and_info, config)
                             for model_and_info in models_and_optimizers]
    # Create the SummaryWriters for Tensorboard
    writers = create_summary_writers(config)
    config.create_dataframe_loggers()

    model = models_and_optimizers[0].model
    optimizer = models_and_optimizers[0].optimizer
    mean_teacher_model = models_and_optimizers[1].model if len(models_and_optimizers) > 1 else None

    # Create LR scheduler
    l_rate_scheduler = LRScheduler(config, optimizer)  # type: ignore

    # Training loop
    logging.info("Starting training")
    train_results_per_epoch, val_results_per_epoch, learning_rates_per_epoch = [], [], []

    last_epoch = config.num_epochs + 1
    resource_monitor = None
    if config.monitoring_interval_seconds > 0:
        # initialize and start GPU monitoring
        resource_monitor = ResourceMonitor(interval_seconds=config.monitoring_interval_seconds,
                                           tb_log_file_path=str(config.logs_folder / "diagnostics"))
        resource_monitor.start()

    for epoch in range(config.start_epoch + 1, last_epoch):
        logging.info("Starting epoch {}".format(epoch))
        # store the learning rates used for each epoch
        epoch_lrs = l_rate_scheduler.get_last_lr()
        learning_rates_per_epoch.append(epoch_lrs)

        train_val_params: TrainValidateParameters = \
            TrainValidateParameters(data_loader=data_loaders[ModelExecutionMode.TRAIN],
                                    model=model,
                                    mean_teacher_model=mean_teacher_model,
                                    epoch=epoch,
                                    optimizer=optimizer,
                                    epoch_learning_rate=epoch_lrs,
                                    summary_writers=writers,
                                    dataframe_loggers=config.metrics_data_frame_loggers,
                                    in_training_mode=True)
        train_epoch_results = train_or_validate_epoch(config, train_val_params)
        train_results_per_epoch.append(train_epoch_results)

        metrics.validate_and_store_model_parameters(writers.train, epoch, model)
        # Run without adjusting weights on the validation set
        train_val_params.in_training_mode = False
        train_val_params.data_loader = data_loaders[ModelExecutionMode.VAL]
        val_epoch_results = train_or_validate_epoch(config, train_val_params)
        val_results_per_epoch.append(val_epoch_results)

        if config.is_segmentation_model:
            metrics.store_epoch_stats_for_segmentation(config.outputs_folder, epoch, epoch_lrs, train_epoch_results,
                                                       val_epoch_results)

        if config.should_save_epoch(epoch) and optimizer is not None:
            save_checkpoint(model, optimizer, epoch, config)
            if config.compute_mean_teacher_model:
                save_checkpoint(mean_teacher_model, optimizer, epoch, config, mean_teacher_model=True)

        # Updating the learning rate should happen at the end of the training loop, so that the
        # initial learning rate will be used for the very first epoch.
        l_rate_scheduler.step()

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

    return ModelTrainingResult(
        train_results_per_epoch=train_results_per_epoch,
        val_results_per_epoch=val_results_per_epoch,
        learning_rates_per_epoch=learning_rates_per_epoch
    )


def train_or_validate_epoch(config: ModelConfigBase,
                            train_val_params: TrainValidateParameters) -> MetricsDict:
    """
    Trains or validates the model for one epoch.
    :param config: The arguments object, which contains useful information for training
    :param train_val_params: The TrainValidateParameters object defining the network and data
    :returns: The results for training or validation. Result type depends on the type of model that is trained.
    """
    epoch_start_time = time()
    training_random_state = None
    if not train_val_params.in_training_mode:
        # take the snapshot of the existing random state
        training_random_state = RandomStateSnapshot.snapshot_random_state()
        # reset the random state for validation
        ml_util.set_random_seed(config.random_seed)
    forward_pass: ModelTrainingStepsBase
    if isinstance(config, SegmentationModelBase):
        forward_pass = ModelTrainingStepsForSegmentation(config, train_val_params)
    elif isinstance(config, ScalarModelBase):
        if isinstance(config, SequenceModelBase):
            forward_pass = ModelTrainingStepsForSequenceModel(config, train_val_params)
        else:
            forward_pass = ModelTrainingStepsForScalarModel(config, train_val_params)
    else:
        raise NotImplementedError(f"There is no forward pass defined for config type {type(config)}")

    status_string = "training" if train_val_params.in_training_mode else "validation"
    item_start_time = time()
    num_load_time_warnings = 0
    num_load_time_exceeded = 0
    num_batches = 0
    total_extra_load_time = 0.0
    total_load_time = 0.0
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
        loss = forward_pass.forward_and_backward_minibatch(sample, batch_index, train_val_params.epoch)
        train_finish_time = time()
        logging.debug(f"Epoch {train_val_params.epoch} {status_string} batch {batch_index}: "
                      f"Loaded in {item_load_time:0.2f}sec, "
                      f"{status_string} in {(train_finish_time - item_finish_time):0.2f}sec. Loss = {loss}")
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
    return forward_pass.get_epoch_results_and_store(epoch_time_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="The name of the model to train", type=empty_string_to_none,
                        required=True)
    args = parser.parse_args()
    model_train(ModelConfigLoader().create_model_config_from_name(args.model))


if __name__ == '__main__':
    main()

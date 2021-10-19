#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import math
from typing import List
from unittest import mock

from InnerEye.ML.lightning_loggers import AzureMLProgressBar, PROGRESS_STAGE_PREDICT, PROGRESS_STAGE_TEST, \
    PROGRESS_STAGE_TRAIN, \
    PROGRESS_STAGE_VAL
from InnerEye.ML.metrics import EpochTimers
from _pytest.logging import LogCaptureFixture


def test_progress_bar_enable() -> None:
    """
    Test the logic for disabling the progress bar.
    """
    bar = AzureMLProgressBar(refresh_rate=0)
    assert not bar.is_enabled
    bar = AzureMLProgressBar(refresh_rate=1)
    assert bar.is_enabled
    bar.disable()
    assert not bar.is_enabled
    bar.enable()
    assert bar.is_enabled


def test_progress_bar() -> None:
    bar = AzureMLProgressBar(refresh_rate=1)
    mock_trainer = mock.MagicMock(current_epoch=12,
                                  lightning_module=mock.MagicMock(global_step=34),
                                  num_training_batches=10,
                                  emable_validation=False,
                                  num_test_batches=[20],
                                  num_predict_batches=[30])
    bar.on_init_end(mock_trainer)  # type: ignore
    assert bar.trainer == mock_trainer
    messages: List[str] = []

    def write_message(message: str) -> None:
        messages.append(message)

    bar.progress_print_fn = write_message
    bar.flush_fn = None
    # Messages in training
    bar.on_train_epoch_start(None, None)  # type: ignore
    assert bar.stage == PROGRESS_STAGE_TRAIN
    assert bar.train_batch_idx == 0
    assert bar.val_batch_idx == 0
    assert bar.test_batch_idx == 0
    assert bar.predict_batch_idx == 0
    bar.on_train_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.train_batch_idx == 1
    assert "Training epoch 12 (step 34)" in messages[-1]
    assert "1/10 ( 10%) completed" in messages[-1]
    # When starting the next training epoch, the counters should be reset
    bar.on_train_epoch_start(None, None)  # type: ignore
    assert bar.train_batch_idx == 0
    # Messages in validation
    bar.on_validation_start(None, None)  # type: ignore
    assert bar.stage == PROGRESS_STAGE_VAL
    assert bar.max_batch_count == 0
    assert bar.val_batch_idx == 0
    # Number of validation batches is difficult to fake, tweak the field where it is stored in the progress bar
    bar.max_batch_count = 5
    bar.on_validation_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.val_batch_idx == 1
    assert "Validation epoch 12: " in messages[-1]
    assert "1/5 ( 20%) completed" in messages[-1]
    # Messages in testing
    bar.on_test_epoch_start(None, None)  # type: ignore
    assert bar.stage == PROGRESS_STAGE_TEST
    test_count = 2
    for _ in range(test_count):
        bar.on_test_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.test_batch_idx == test_count
    assert "Testing:" in messages[-1]
    assert f"{test_count}/20 ( 10%)" in messages[-1]
    # Messages in prediction
    bar.on_predict_epoch_start(None, None)  # type: ignore
    assert bar.stage == PROGRESS_STAGE_PREDICT
    predict_count = 3
    for _ in range(predict_count):
        bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.predict_batch_idx == predict_count
    assert "Prediction:" in messages[-1]
    assert f"{predict_count}/30 ( 10%)" in messages[-1]
    # Test behaviour when a batch count is infinity
    bar.max_batch_count = math.inf
    bar.on_predict_batch_end(None, None, None, None, None, None)  # type: ignore
    assert bar.predict_batch_idx == 4
    assert "4 batches completed" in messages[-1]


def test_epoch_timers(caplog: LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    batch_index = 123
    epoch = 24
    timer = EpochTimers(max_item_load_time_seconds=100)
    assert timer.total_load_time == 0.0

    # First batch should always generate a message
    timer.batch_start(batch_index=0, epoch=epoch, message_prefix="prefix")
    assert timer.total_load_time > 0.0
    message = caplog.messages[-1]
    assert "prefix: Loaded the first minibatch of data in" in message
    old_num_batches = timer.num_batches
    old_batch_start_time = timer.batch_start_time
    timer.batch_end()
    assert timer.num_batches == old_num_batches + 1
    assert timer.batch_start_time > old_batch_start_time

    # Second minibatch should only generate a message if above load time threshold. Set threshold very high
    old_num_messages = len(caplog.messages)
    old_total_load_time = timer.total_load_time
    timer.max_item_load_time_seconds = 10.0
    assert timer.num_load_time_exceeded == 0
    timer.batch_start(batch_index=batch_index, epoch=epoch, message_prefix="prefix")
    # This should be updated in any case
    assert timer.total_load_time > old_total_load_time
    # But this batch should not be recognized as having gone over the threshold
    assert timer.num_load_time_exceeded == 0
    assert len(timer.load_time_warning_epochs) == 0
    assert len(caplog.messages) == old_num_messages
    assert timer.num_load_time_warnings == 0

    # Third minibatch considered as above threshold: set threshold to 0 for that
    old_total_load_time = timer.total_load_time
    timer.max_item_load_time_seconds = 0.0
    timer.batch_start(batch_index=batch_index, epoch=epoch, message_prefix="prefix")
    # This should be updated in any case
    assert timer.total_load_time > old_total_load_time
    # Batch should not be recognized as having gone over the threshold
    assert timer.num_load_time_exceeded == 1
    assert epoch in timer.load_time_warning_epochs
    message = caplog.messages[-1]
    assert f"prefix: Loading minibatch { batch_index} took" in message
    assert f"This message will be printed at most {timer.max_load_time_warnings} times"
    assert timer.num_load_time_warnings > 0

    # Epoch end time should be stored
    assert timer.total_epoch_time == 0.0
    old_epoch_end_time = timer.epoch_end_time
    timer.epoch_end()
    assert timer.epoch_end_time > old_epoch_end_time
    assert timer.total_epoch_time > 0.0

    timer.reset()
    assert timer.total_load_time == 0.0
    assert timer.num_load_time_warnings == 0

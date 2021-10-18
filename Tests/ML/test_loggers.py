#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
from typing import List
from unittest import mock

from InnerEye.ML.lightning_loggers import AzureMLProgressBar, PROGRESS_STAGE_PREDICT, PROGRESS_STAGE_TEST, \
    PROGRESS_STAGE_TRAIN, \
    PROGRESS_STAGE_VAL


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


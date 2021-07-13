#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import List

import torch

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.other.DummyEnsembleRegressionContainer import DummyEnsembleRegressionModule
from InnerEye.ML.configs.other.HelloContainer import HelloContainer, HelloDataModule


def test_random_ensemble(test_output_dirs: OutputFolderForTests) -> None:
    """
    Make dummy checkpoints and load them, test ensemble behaves reasonably.

    N.B. we cannot use create_model_and_store_checkpoint as its call to create_lightning_model fails with an
    AttributeError: 'DummyEnsembleRegressionContainer' object has no attribute 'is_segmentation_model'
    """
    # Create checkpoints from initial random weights (i.e. without training)
    cross_validation_children: List[DummyEnsembleRegressionModule] = []
    checkpoint_paths: List[Path] = []
    for idx in range(5):
        checkpoint_path = test_output_dirs.root_dir / f"{idx}.ckpt"
        module = DummyEnsembleRegressionModule(outputs_folder=test_output_dirs.root_dir)
        torch.save({'state_dict': module.model.state_dict()}, checkpoint_path)
        cross_validation_children.append(module)
        checkpoint_paths.append(checkpoint_path)
    # Load checkpoints as ensemble
    eldest = cross_validation_children[0]
    eldest.load_checkpoints_as_siblings(checkpoint_paths, use_gpu=False)
    # Get test data split
    data_module_xval = HelloDataModule(
        root_folder=HelloContainer().local_dataset,  # type: ignore
        cross_validation_split_index=0,
        number_of_cross_validation_splits=5)
    test_dataloader = data_module_xval.test_dataloader()
    # Run inference loop
    eldest.on_inference_start()
    eldest.on_inference_start_dataset(execution_mode=ModelExecutionMode.TEST, _=True)
    for batch_idx, batch in enumerate(test_dataloader):
        posteriors = eldest.forward(batch['x'])
        eldest.record_posteriors(batch, batch_idx, posteriors)
    eldest.on_inference_end_dataset()
    xval_metrics_dir = test_output_dirs.root_dir / str(ModelExecutionMode.TEST)
    assert (xval_metrics_dir / "test_mse.txt").exists
    assert (xval_metrics_dir / "test_mae.txt").exists

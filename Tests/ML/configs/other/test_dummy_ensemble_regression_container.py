#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest
from unittest import mock

from pytorch_lightning.trainer import configuration_validator
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.configs.other.HelloContainer import HelloDataModule, HelloDataset, HelloContainer, HelloRegression
from InnerEye.ML.configs.other.DummyEnsembleRegressionContainer import DummyEnsembleRegressionModule, DummyEnsembleRegressionContainer
from Tests.ML.util import default_runner
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint, create_lightning_model
from InnerEye.ML.model_training import create_lightning_trainer

def test_attempt_1(test_output_dirs: OutputFolderForTests) -> None:
    """
    Make dummy checkpoints and load them, test ensemble behaves reasonably
    """
    # Make checkpoints
    checkpoint_paths = []
    for idx in range(5):
        checkpoint_path = test_output_dirs.root_dir / f"{idx}.ckpt"
        # FAILS HERE with error from first line of create_model_and_store_checkpoint when:
        # create_lightning_model(config)
        # AttributeError: 'DummyEnsembleRegressionContainer' object has no attribute 'is_segmentation_model'
        create_model_and_store_checkpoint(
            config=DummyEnsembleRegressionContainer(),
            checkpoint_path=checkpoint_path,
            weights_only=False)
        checkpoint_paths.append(checkpoint_path)
    # Create module and load checkpoints as ensemble
    ensemble_module = DummyEnsembleRegressionModule(outputs_folder=test_output_dirs)
    ensemble_module.load_checkpoints_as_sibling(
        paths_to_checkpoints=checkpoint_paths,
        use_gpu=False)
    # Get test data split
    data_module_xval = HelloDataModule(
        root_folder=HelloContainer().local_dataset,
        cross_validation_split_index=0,
        number_of_cross_validation_splits=5)
    test_dataloader = data_module_xval.test_dataloader
    # Run inference loop
    ensemble_module.on_inference_start()
    ensemble_module.on_inference_start_dataset(execution_mode=ModelExecutionMode.TEST, _=True)
    for batch_idx, batch in enumerate(test_dataloader):
        posteriors = ensemble_module.forward(batch)
        ensemble_module.record_posteriors(batch, batch_idx, posteriors)
    ensemble_module.on_inference_end_dataset()
    # Assert something

def test_checkpoint_handling(test_output_dirs: OutputFolderForTests) -> None:
    """
    """
    checkpoint_paths = []
    for idx in range(5):
        checkpoint_path = test_output_dirs.root_dir / f"{idx}.ckpt"
        container = DummyEnsembleRegressionContainer()
        trainer, _ = create_lightning_trainer(container=container)
        trainer.model = create_lightning_model()
        create_model_and_store_checkpoint(
            config=DummyEnsembleRegressionContainer,
            checkpoint_path=checkpoint_path,
            weights_only=False)
        checkpoint_paths.append(checkpoint_path)
    ensemble_module = DummyEnsembleRegressionModule(outputs_folder=test_output_dirs)
    ensemble_module.load_checkpoints_as_sibling(
        paths_to_checkpoints=checkpoint_paths,
        use_gpu=False)
    data_module_xval = HelloDataModule(
        root_folder=HelloContainer().local_dataset,
        cross_validation_split_index=0,
        number_of_cross_validation_splits=5)
    test_dataloader = data_module_xval.test_dataloader
    ensemble_module.on_inference_start()
    ensemble_module.on_inference_start_dataset(execution_mode=ModelExecutionMode.TEST, _=True)
    for batch_idx, batch in enumerate(test_dataloader):
        posteriors = ensemble_module.forward(batch)
        ensemble_module.record_posteriors(batch, batch_idx, posteriors)
    ensemble_module.on_inference_end_dataset()


# def test_local_cross_validation_training_then_inference(test_output_dirs: OutputFolderForTests) -> None:
#     """
#     Temporary test so I can work through the implementation of cross validation in DummyEnsembleRegressionModule
#     """

    # runner = default_runner()
    # dataset_dir = test_output_dirs.root_dir / "dataset"
    # dataset_dir.mkdir(parents=True)
    # args = ["", "--model=DummyEnsembleRegressionContainer",
    #         f"--output_to={test_output_dirs.root_dir}",
    #         "--model_configs_namespace=Tests.ML.configs"]
    # with mock.patch("sys.argv", args):
    #     parser_result = runner.parse_and_load_model()
    # ml_runner = runner.create_ml_runner()
    # ml_runner.setup()
    # checkpoint_path = ml_runner.checkpoint_handler.get_recovery_path_train()
    # lightning_model = ml_runner.container.model
    # trainer, storing_logger = create_lightning_trainer(container,
    #                                                 checkpoint_path,
    #                                                 num_nodes=num_nodes,
    #                                                 **container.get_trainer_arguments())


    # from Tests.ML.configs.fastmri_random import FastMriOnRandomData
    # assert isinstance(runner.lightning_container, FastMriOnRandomData)

    # for cross_validation_split_index in range(5):
    #     data_module_xval = DummyEnsembleRegressionModule(
    #         outputs_folder=test_output_dirs.root_dir,
    #         cross_validation_split_index=cross_validation_split_index,
    #         number_of_cross_validation_splits=5)
    #     data_module_xval.train()

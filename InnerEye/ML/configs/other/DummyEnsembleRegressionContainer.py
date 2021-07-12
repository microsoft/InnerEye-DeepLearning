#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Optional

import torch
from pytorch_lightning.metrics import MeanAbsoluteError

from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.other.HelloContainer import HelloContainer, HelloRegression
from InnerEye.ML.deep_learning_config import load_checkpoint
from InnerEye.ML.lightning_base import LightningModule
from InnerEye.ML.lightning_container import InnerEyeInference


class DummyEnsembleRegressionModule(HelloRegression, InnerEyeInference):
    """
    A simple 1-dim regression model intended to be trained across cross validation splits and then evaluated on test
    data using an ensemble built from the checkpoints coming from from the cross validation splits.

    To perform the ensemble inference we need to implement the methods from InnerEyeInference.
    """

    def __init__(self, outputs_folder: Path) -> None:
        HelloRegression.__init__(self)
        self.outputs_folder = outputs_folder
        self.siblings: List[DummyEnsembleRegressionModule] = [self]
        self.test_mse: List[torch.Tensor] = []
        self.test_mae = MeanAbsoluteError()
        self.execution_mode: Optional[ModelExecutionMode] = None

    def load_checkpoints_as_siblings(self, paths_to_checkpoints: List[Path], use_gpu: bool) -> None:
        """
        Load a set of checkpoints as siblings, for example the set of checkpoints from cross validation runs used to
        make an ensemble model.
        :param paths_to_checkpoints: A list of paths to checkpoint files.
        :param use_gpu: Use the GPU or map to the CPU? This is known by the container, but not at this, the data module
        level.
        """
        if len(paths_to_checkpoints) < 1:
            raise ValueError("The list of paths to checkpoints must include at least one path.")
        checkpoint = load_checkpoint(paths_to_checkpoints[0], use_gpu)
        self.load_state_dict(checkpoint['state_dict'])
        for path_to_checkpoint in paths_to_checkpoints[1:]:
            self.load_checkpoint_as_sibling(path_to_checkpoint, use_gpu)

    def load_checkpoint_as_sibling(self, path_to_checkpoint: Path, use_gpu: bool) -> None:
        """
        Load a sibling from its checkpoint, for example a later child from a cross validation runs used to make an
        ensemble model.
        :param path_to_checkpoint: The path to the checkpoint file.
        :param use_gpu: Use the GPU or map to the CPU? This is known by the container, but not at this, the data module
        level.
        """
        checkpoint = load_checkpoint(path_to_checkpoint, use_gpu)
        sibling = DummyEnsembleRegressionModule(self.outputs_folder)
        sibling.load_state_dict(checkpoint['state_dict'])
        self.siblings.append(sibling)
        self.inference_losses: List[List[List[torch.Tensor]]] = []  # losses for each epoch, sibling, and step
        self.epoch_count = 0

    # region InnerEyeInference Overrides
    def on_inference_start(self) -> None:
        """
        Runs initialization for everything that inference might require. This can initialize
        output files, set up metric computation, etc. This is run only once.
        """
        for sibling in self.siblings:
            sibling.eval()
        self.execution_mode = None

    def on_inference_start_dataset(self, execution_mode: ModelExecutionMode, _: bool) -> None:
        """
        Runs initialization for inference, when starting inference on a new dataset split (train/val/test).
        Depending on the settings, this can be called anywhere between 0 (no inference at all) to 3 times (inference
        on all of train/val/test split).
        :param execution_mode: Indicates whether the item comes from the training, validation or test set.
        """
        # We will use the HelloRegression test provision, regardless of the actual ModelExecutionMode specified for the
        # inference.
        for sibling in self.siblings:
            sibling.on_test_epoch_start()
        self.test_mse = []
        self.test_mae.reset()
        self.execution_mode = execution_mode

    def record_posteriors(self, batch: Dict[str, torch.Tensor], batch_idx: int, _: torch.Tensor) -> None:
        """
        This hook is called when the model has finished making a prediction. It can write the results to a file,
        or compute metrics and store them.
        :param batch: The batch of data for which the model made a prediction.
        """
        posteriors: List[torch.Tensor] = []
        target = batch["y"]
        for sibling in self.siblings:
            posteriors.append(sibling.test_step(batch, batch_idx))
        posterior = InnerEyeInference.aggregate_ensemble_model_outputs(iter(posteriors))
        loss = torch.nn.functional.mse_loss(posterior, target)
        self.test_mse.append(loss)
        self.test_mae.update(preds=posterior, target=target)

    def on_inference_end_dataset(self) -> None:
        """
        Write the metrics from the inference execution to disk
        We do not call HelloRegression.on_test_epoch_end since we handle the writing to disk.
        """
        output_dir = self.outputs_folder / str(self.execution_mode)
        average_mse = torch.mean(torch.stack(self.test_mse))
        with (output_dir / "test_mse.txt").open("a") as test_mse_file:
            test_mse_file.write(f"Epoch {self.epoch_count + 1}: {average_mse.item()}\n")
        with (output_dir / "test_mae.txt").open("a") as test_mae_file:
            test_mae_file.write(f"Epoch {self.epoch_count + 1}: {str(self.test_mae.compute())}\n")
    # endregion

    # region HelloRegression Overrides
    def test_step(self, batch: Dict[str, torch.Tensor], _: int) -> torch.Tensor:  # type: ignore
        """
        The HelloRegression version only returns the loss, not the prediction, which we need for ensemble calculations.
        :param batch: The batch of test data.
        :param batch_idx: The index (0, 1, ...) of the batch when the data loader is enumerated.
        :return: The posterior from the test data.
        """
        input = batch["x"]
        return self.forward(input)
    # endregion

class DummyEnsembleRegressionContainer(HelloContainer):
    """
    Exemplar class, based on the simple linear regression model HelloContainer, designed to show how to enable building
    an ensemble model from the checkpoints of a BYOL (Bring Your Own Lightning) cross validation model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.number_of_cross_validation_splits = 5

    def create_model(self) -> LightningModule:
        return DummyEnsembleRegressionModule(outputs_folder=self.outputs_folder)

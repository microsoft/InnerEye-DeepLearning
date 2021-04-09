#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import param
import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.metrics import MeanSquaredError
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import DataLoader, Dataset

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference


class DummyContainerWithDatasets(LightningContainer):
    def __init__(self, has_local_dataset: bool = False, has_azure_dataset: bool = False):
        super().__init__()
        self.local_dataset = full_ml_test_data_path("lightning_module_data") if has_local_dataset else None
        self.azure_dataset_id = "azure_dataset" if has_azure_dataset else ""

    def create_model(self) -> LightningWithInference:
        return LightningWithInference()


class DummyContainerWithAzureDataset(DummyContainerWithDatasets):
    def __init__(self):
        super().__init__(has_azure_dataset=True)


class DummyContainerWithoutDataset(DummyContainerWithDatasets):
    pass


class DummyContainerWithLocalDataset(DummyContainerWithDatasets):
    def __init__(self):
        super().__init__(has_local_dataset=True)


class DummyContainerWithAzureAndLocalDataset(DummyContainerWithDatasets):
    def __init__(self):
        super().__init__(has_local_dataset=True, has_azure_dataset=True)


class InferenceWithParameters(LightningWithInference):
    model_param = param.String(default="bar")

    def __init__(self, container_param: str):
        super().__init__()


class DummyContainerWithParameters(LightningContainer):
    container_param = param.String(default="foo")

    def __init__(self):
        super().__init__()

    def create_model(self) -> LightningWithInference:
        return InferenceWithParameters(self.container_param)


class DummyRegressionPlainLightning(LightningWithInference):
    """
    A class that only implements plain Lightning training and test. Ideally, we want to support importing any plain
    Lightning module without further methods added. This class here inherits LightningWithInference, but does not
    implement the inference_step method
    """

    def __init__(self, in_features: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.l_rate = 1e-1
        activation = Identity()
        layers = [
            torch.nn.Linear(in_features=in_features, out_features=1, bias=True),
            activation
        ]
        self.model = torch.nn.Sequential(*layers)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        Path("test_step.txt").touch()
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("test_loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_test_epoch_end(self) -> None:
        Path("on_test_epoch_end.txt").touch()
        pass


class DummyRegression(DummyRegressionPlainLightning):
    def __init__(self, in_features: int = 1, *args, **kwargs):
        super().__init__(in_features=in_features, *args, **kwargs)
        self.l_rate = 1e-1
        self.dataset_split = ModelExecutionMode.TRAIN
        activation = Identity()
        layers = [
            torch.nn.Linear(in_features=in_features, out_features=1, bias=True),
            activation
        ]
        self.model = torch.nn.Sequential(*layers)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch, *args, **kwargs):
        input, target = batch
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("loss", loss, on_epoch=True, on_step=True)
        return loss

    def on_inference_start(self) -> None:
        (self.outputs_folder / "on_inference_start.txt").touch()
        self.inference_mse: Dict[ModelExecutionMode, float] = {}

    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        self.dataset_split = dataset_split
        (self.outputs_folder / f"on_inference_start_{self.dataset_split.value}.txt").touch()
        self.mse = MeanSquaredError()

    def inference_step(self, item: Tuple[Tensor, Tensor], batch_idx, **kwargs):
        input, target = item
        prediction = self.forward(input)
        self.mse(prediction, target)
        with (self.outputs_folder / f"inference_step_{self.dataset_split.value}.txt").open(mode="a") as f:
            f.write(f"{prediction.item()},{target.item()}\n")

    def on_inference_epoch_end(self) -> None:
        (self.outputs_folder / f"on_inference_end_{self.dataset_split.value}.txt").touch()
        self.inference_mse[self.dataset_split] = self.mse.compute().item()
        self.mse.reset()

    def on_inference_end(self) -> None:
        (self.outputs_folder / "on_inference_end.txt").touch()
        df = pd.DataFrame(columns=["Split", "MSE"],
                          data=[[split.value, mse] for split, mse in self.inference_mse.items()])
        df.to_csv(self.outputs_folder / "metrics_per_split.csv", index=False)


class FixedDataset(Dataset):
    def __init__(self, inputs_and_targets: List[Tuple]):
        super().__init__()
        self.inputs_and_targets = inputs_and_targets

    def __len__(self) -> int:
        return len(self.inputs_and_targets)

    def __getitem__(self, item: int) -> Tuple[Tensor, Tensor]:
        input = torch.tensor([float(self.inputs_and_targets[item][0])])
        target = torch.tensor([float(self.inputs_and_targets[item][1])])
        return input, target


class FixedRegressionData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.train_data = [(i, i) for i in range(1, 20, 3)]
        self.val_data = [(i, i) for i in range(2, 20, 3)]
        self.test_data = [(i, i) for i in range(3, 20, 3)]

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(FixedDataset(self.train_data))

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(FixedDataset(self.val_data))

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(FixedDataset(self.test_data))


class DummyContainerWithModel(LightningContainer):

    def __init__(self):
        super().__init__()
        self.perform_training_set_inference = True
        self.num_epochs = 50
        self.l_rate = 1e-1

    def setup(self) -> None:
        (self.local_dataset / "setup.txt").touch()

    def create_model(self) -> LightningWithInference:
        return DummyRegression()

    def get_data_module(self) -> LightningDataModule:
        return FixedRegressionData()


class DummyContainerWithInvalidTrainerArguments(DummyContainerWithModel):
    def get_trainer_arguments(self):
        return {"no_such_argument": 1}


class DummyContainerWithPlainLightning(LightningContainer):
    def __init__(self):
        super().__init__()
        self.num_epochs = 100
        self.l_rate = 1e-2

    def create_model(self) -> LightningWithInference:
        return DummyRegressionPlainLightning()

    def get_data_module(self) -> LightningDataModule:
        return FixedRegressionData()

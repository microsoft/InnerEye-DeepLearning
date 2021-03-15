#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.nn import Identity
from torch.utils.data import DataLoader, Dataset

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference


class DummyContainerWithAzureDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(azure_dataset_id="azure_dataset", local_dataset=local_dataset)


class DummyContainerWithoutDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        return LightningWithInference()


class DummyContainerWithLocalDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(local_dataset=local_dataset)


class DummyContainerWithAzureAndLocalDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(azure_dataset_id="azure_dataset", local_dataset=local_dataset)


class DummyRegression(LightningWithInference):
    def __init__(self, in_features: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return loss


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

    def create_lightning_module(self) -> LightningWithInference:
        return DummyRegression()

    def get_training_data_module(self, crossval_index: int, crossval_count: int) -> LightningDataModule:
        return FixedRegressionData()

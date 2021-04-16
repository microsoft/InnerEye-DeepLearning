#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset

from InnerEye.Common import fixed_paths_for_tests
from InnerEye.ML.lightning_container import LightningContainer


class HelloDataset(Dataset):
    def __init__(self, root_folder: Path, start_index: int, end_index: int) -> None:
        self.data = torch.tensor(np.loadtxt(root_folder / "hellocontainer.csv", delimiter=",")[start_index:end_index])

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item) -> Dict[str, torch.Tensor]:
        return {'x': self.data[item][0], 'y': self.data[item][1]}


class HelloDataModule(LightningDataModule):
    def __init__(self, root_folder: Path) -> None:
        self.train = HelloDataset(root_folder, start_index=0, end_index=50)
        self.val = HelloDataset(root_folder, start_index=50, end_index=70)
        self.test = HelloDataset(root_folder, start_index=70, end_index=100)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train, batch_size=5)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.val, batch_size=5)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test, batch_size=5)


class HelloRegression(LightningModule):
    def __init__(self, in_features: int = 1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.l_rate = 1e-1
        self.model = torch.nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.test_mse = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def training_step(self, batch: Any, *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        optimizer = Adam(self.parameters(), lr=1e-2)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def on_test_epoch_start(self) -> None:
        self.test_mse = []

    def test_step(self, batch, batch_idx) -> torch.Tensor:  # type: ignore
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.test_mse.append(loss)
        return loss

    def on_test_epoch_end(self) -> None:
        average_mse = torch.mean(torch.cat(self.test_mse))
        Path("test_mse.txt").write_text(str(average_mse.item()))


class HelloContainer(LightningContainer):

    def __init__(self):
        self.local_dataset = fixed_paths_for_tests.full_ml_test_data_path()
        self.num_epochs = 20

    def create_model(self) -> LightningModule:
        return HelloRegression()

    def get_data_module(self) -> LightningDataModule:
        return HelloDataModule(root_folder=self.local_dataset)

    def create_report(self) -> None:
        # This just prints out the test MSE, but you could also generate a Jupyter notebook here, for example.
        test_mse = float(Path("test_mse.txt").read_text())
        report = f"Performance on test set: MSE = {test_mse}"
        print(report)
        Path("report.txt").write_text(report)

# import numpy as np
# import torch
#
# N = 100
# x = torch.rand((N, 1)) * 10
# y = 0.2 * x + 0.1 * torch.randn(x.size())
# xy = torch.cat((x, y), dim=1)
# np.savetxt("Tests/ML/test_data/hellocontainer.csv", xy.numpy(), delimiter=",")

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset as TorchDataset, DataLoader

logging.getLogger().setLevel(logging.INFO)


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(in_features=1, out_features=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)


class LinearRegressionDataset(TorchDataset):
    def __init__(self, filename: Path, x_col: str, y_col: str):
        df = pd.read_csv(filename)
        xs = df[x_col][:].values
        ys = df[y_col][:].values

        self.x = torch.from_numpy(xs.reshape(-1, 1).astype(np.float32))
        self.y = torch.from_numpy(ys.reshape(-1, 1).astype(np.float32))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.x)


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module, optimizer: Optimizer) -> None:
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            logging.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def ml_test_loop(dataloader: DataLoader, model: nn.Module, loss_fn: nn.Module) -> float:
    size = len(dataloader.dataset)
    test_loss, correct = 0., 0.

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    logging.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss

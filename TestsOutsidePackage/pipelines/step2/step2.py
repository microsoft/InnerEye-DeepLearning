#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import pandas as pd
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split


logging.getLogger().setLevel(logging.INFO)


class LinearRegressionModel(nn.Module):
    def __init__(self):
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

    def __len__(self):
        return len(self.x)


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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


def ml_test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size
    logging.info(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def test_linear_regression(csv_file_path: Path) -> nn.Module:
    dataset = LinearRegressionDataset(csv_file_path, 'xs', 'ys')

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    model = LinearRegressionModel()

    learning_rate = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    epochs = 200

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        ml_test_loop(test_dataloader, model, loss_fn)

    return model


def step2(input_folder: str, input_file: str, output_folder: str, output_file: str) -> None:
    input_file_path = Path(input_folder) / input_file

    model = test_linear_regression(input_file_path)

    state_dict = model.state_dict()
    logging.info("model state_dict: %s", state_dict)

    new_variable = torch.Tensor([[6.0]])
    prediction_y = model(new_variable)
    logging.info("prediction for value %s, %s", 6, prediction_y.data[0][0])

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    torch.save(state_dict, output_path / output_file)


def main() -> None:
    logging.info("in main")

    parser = argparse.ArgumentParser("step2")
    parser.add_argument("--input_step2_folder", type=str, help="input_step2 folder")
    parser.add_argument("--input_step2_file", type=str, help="input_step2 file")
    parser.add_argument("--output_step2_folder", type=str, help="output_step2 folder")
    parser.add_argument("--output_step2_file", type=str, help="output_step2 file")

    args = parser.parse_args()
    logging.info("args: %s", args)

    step2(args.input_step2_folder,
          args.input_step2_file,
          args.output_step2_folder,
          args.output_step2_file)


if __name__ == "__main__":
    logging.info("in wrapper")
    main()

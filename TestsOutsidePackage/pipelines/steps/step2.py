#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
from pathlib import Path

from azureml.core.run import Run
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, random_split

from model.regression import LinearRegressionDataset, LinearRegressionModel, train_loop, ml_test_loop

logging.getLogger().setLevel(logging.INFO)


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

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    run = Run.get_context()

    for t in range(epochs):
        logging.info(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        loss = ml_test_loop(test_dataloader, model, loss_fn)
        run.log('loss', loss)
        run.parent.log('train_loss', loss)

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

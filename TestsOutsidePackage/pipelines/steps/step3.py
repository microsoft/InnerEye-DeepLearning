#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.regression import LinearRegressionModel

logging.getLogger().setLevel(logging.INFO)


def step3(input_data_folder: str, input_data_file: str, input_model_folder: str, input_model_file: str,
          output_folder: str, output_file: str) -> None:

    input_data_file_path = Path(input_data_folder) / input_data_file
    input_model_file_path = Path(input_model_folder) / input_model_file

    model = LinearRegressionModel()
    model.load_state_dict(torch.load(input_model_file_path))

    df = pd.read_csv(input_data_file_path)

    logging.info("df_in")
    logging.info(df.to_string(max_rows=20))

    xs = df['xs'][:].values
    xs = torch.from_numpy(xs.reshape(-1, 1).astype(np.float32))

    model.eval()
    with torch.no_grad():
        pred = model(xs)

    df["ys"] = pred

    logging.info("df_out")
    logging.info(df.to_string(max_rows=20))

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    df.to_csv(output_path / output_file)


def main() -> None:
    logging.info("in main")

    parser = argparse.ArgumentParser("step3")
    parser.add_argument("--input_step3_data_folder", type=str, help="input_step3 data folder")
    parser.add_argument("--input_step3_data_file", type=str, help="input_step3 data file")
    parser.add_argument("--input_step3_model_folder", type=str, help="input_step3 model folder")
    parser.add_argument("--input_step3_model_file", type=str, help="input_step3 model file")
    parser.add_argument("--output_step3_folder", type=str, help="output_step3 folder")
    parser.add_argument("--output_step3_file", type=str, help="output_step3 file")

    args = parser.parse_args()
    logging.info("args: %s", args)

    step3(args.input_step3_data_folder,
          args.input_step3_data_file,
          args.input_step3_model_folder,
          args.input_step3_model_file,
          args.output_step3_folder,
          args.output_step3_file)


if __name__ == "__main__":
    logging.info("in wrapper")
    main()

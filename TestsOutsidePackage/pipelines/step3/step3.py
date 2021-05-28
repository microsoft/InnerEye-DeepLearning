#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import pandas as pd
from pathlib import Path

logging.getLogger().setLevel(logging.INFO)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    xs = df['x'][:].values
    ys = df['y'][:].values

    xs = xs + 3
    ys = ys + 4.25

    return pd.DataFrame({'x': xs, 'y': ys})


def step3(input_folder: str, input_file: str, output_folder: str, output_file: str) -> None:
    input_file_path = Path(input_folder) / input_file

    df_in = pd.read_csv(input_file_path)

    logging.info("df_in")
    logging.info(df_in.to_string(max_rows=20))

    df_out = preprocess(df_in)

    logging.info("df_out")
    logging.info(df_out.to_string(max_rows=20))

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)

    df_out.to_csv(output_path / output_file)


def main() -> None:
    logging.info("in main")

    parser = argparse.ArgumentParser("step3")
    parser.add_argument("--input_step3_folder", type=str, help="input_step3 folder")
    parser.add_argument("--input_step3_file", type=str, help="input_step3 file")
    parser.add_argument("--output_step3_folder", type=str, help="output_step3 folder")
    parser.add_argument("--output_step3_file", type=str, help="output_step3 file")

    args = parser.parse_args()
    logging.info("args: %s", args)

    step3(args.input_step3_folder,
          args.input_step3_file,
          args.output_step3_folder,
          args.output_step3_file)


if __name__ == "__main__":
    logging.info("in wrapper")
    main()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.utils.dataset_util import add_label_stats_to_dataframe


def main(args: Any) -> None:
    dataframe_path = args.data_root_dir / Path(DATASET_CSV_FILE_NAME)
    dataframe = add_label_stats_to_dataframe(input_dataframe=pd.read_csv(str(dataframe_path)),
                                             dataset_root_directory=args.data_root_dir,
                                             target_label_names=args.target_label_names)

    # Write the dataframe in a file and exit
    dataframe.to_csv(args.output_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # noinspection PyTypeChecker
    parser.add_argument("--data-root-dir", type=Path, help="Path to data root directory")
    parser.add_argument("--target-label-names", nargs='+', type=str, help="Names of target structures e.g. prostate")
    # noinspection PyTypeChecker
    parser.add_argument("--output-csv-path", type=Path, help="Path to output csv file")
    args = parser.parse_args()

    # Sample run
    # python check_annotation_quality.py
    #   --data-root-dir "/path/to/your/data/directory"
    #   --target-label-names femur_r femur_l rectum prostate bladder seminalvesicles
    #   --output-csv-path "/path/to/your/data/directory/label_stats.csv"

    # Dataset blob folder
    main(args)

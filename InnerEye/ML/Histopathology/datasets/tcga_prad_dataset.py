#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union

import pandas as pd

from InnerEye.ML.Histopathology.datasets.base_dataset import SlidesDataset


class TcgaPradDataset(SlidesDataset):
    """Dataset class for loading TCGA-PRAD slides.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str)
    - `'case_id'` (str)
    - `'image_path'` (str): absolute slide image path
    - `'label'` (int, 0 or 1): label for predicting positive or negative
    """
    IMAGE_COLUMN: str = 'image_path'
    LABEL_COLUMN: str = 'label'

    DEFAULT_CSV_FILENAME: str = "dataset.csv"

    def __init__(self, root: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file. If omitted, the CSV will be read from
        `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        """
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False)
        self.dataset_df[self.LABEL_COLUMN] = (self.dataset_df['label1_mutation']
                                              | self.dataset_df['label2_mutation']).astype(int)
        self.validate_columns()

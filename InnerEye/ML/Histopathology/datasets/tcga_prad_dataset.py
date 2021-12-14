#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from torch.utils.data import Dataset


class TcgaPradDataset(Dataset):
    """Dataset class for loading TCGA-PRAD slides.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str)
    - `'case_id'` (str)
    - `'image_path'` (str): absolute slide image path
    - `'label'` (int, 0 or 1): label for predicting positive or negative
    """
    SLIDE_ID_COLUMN: str = 'slide_id'
    CASE_ID_COLUMN: str = 'case_id'
    IMAGE_COLUMN: str = 'image_path'
    LABEL_COLUMN: str = 'label'

    DEFAULT_CSV_FILENAME: str = "dataset.csv"

    def __init__(self, root_dir: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file. If omitted, the CSV will be read from
        `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        """
        self.root_dir = Path(root_dir)

        if dataset_df is not None:
            self.dataset_csv = None
        else:
            self.dataset_csv = dataset_csv or self.root_dir / self.DEFAULT_CSV_FILENAME
            dataset_df = pd.read_csv(self.dataset_csv)

        dataset_df = dataset_df.set_index(self.SLIDE_ID_COLUMN)
        dataset_df[self.LABEL_COLUMN] = (dataset_df['label1_mutation']
                                         | dataset_df['label2_mutation']).astype(int)
        self.dataset_df = dataset_df

    def __len__(self) -> int:
        return self.dataset_df.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        slide_id = self.dataset_df.index[index]
        sample = {
            self.SLIDE_ID_COLUMN: slide_id,
            **self.dataset_df.loc[slide_id].to_dict()
        }
        sample[self.IMAGE_COLUMN] = str(self.root_dir / sample.pop(self.IMAGE_COLUMN))
        return sample

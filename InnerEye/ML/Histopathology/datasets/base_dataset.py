from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset


class TilesDataset(Dataset):
    """Base class for datasets of WSI tiles, iterating dictionaries of image paths and metadata.

    :param TILE_ID_COLUMN: CSV column name for tile ID.
    :param SLIDE_ID_COLUMN: CSV column name for slide ID.
    :param IMAGE_COLUMN: CSV column name for relative path to image file.
    :param PATH_COLUMN: CSV column name for relative path to image file. Replicated to propagate the path to the batch.
    :param LABEL_COLUMN: CSV column name for tile label.
    :param SPLIT_COLUMN: CSV column name for train/test split (optional).
    :param TILE_X_COLUMN: CSV column name for horizontal tile coordinate (optional).
    :param TILE_Y_COLUMN: CSV column name for vertical tile coordinate (optional).
    :param TRAIN_SPLIT_LABEL: Value used to indicate the training split in `SPLIT_COLUMN`.
    :param TEST_SPLIT_LABEL: Value used to indicate the test split in `SPLIT_COLUMN`.
    :param DEFAULT_CSV_FILENAME: Default name of the dataset CSV at the dataset rood directory.
    :param N_CLASSES: Number of classes indexed in `LABEL_COLUMN`.
    """
    TILE_ID_COLUMN: str = 'tile_id'
    SLIDE_ID_COLUMN: str = 'slide_id'
    IMAGE_COLUMN: str = 'image'
    PATH_COLUMN: str = 'image_path'
    LABEL_COLUMN: str = 'label'
    SPLIT_COLUMN: Optional[str] = 'split'
    TILE_X_COLUMN: Optional[str] = 'tile_x'
    TILE_Y_COLUMN: Optional[str] = 'tile_y'

    TRAIN_SPLIT_LABEL: str = 'train'
    TEST_SPLIT_LABEL: str = 'test'

    DEFAULT_CSV_FILENAME: str = "dataset.csv"

    N_CLASSES: int = 1  # binary classification by default

    def __init__(self,
                 root: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 train: Optional[bool] = None) -> None:
        """
        :param root: Root directory of the dataset.
        :param dataset_csv: Full path to a dataset CSV file, containing at least
        `TILE_ID_COLUMN`, `SLIDE_ID_COLUMN`, and `IMAGE_COLUMN`. If omitted, the CSV will be read
        from `"{root}/{DEFAULT_CSV_FILENAME}"`.
        :param dataset_df: A potentially pre-processed dataframe in the same format as would be read
        from the dataset CSV file, e.g. after some filtering. If given, overrides `dataset_csv`.
        :param train: If `True`, loads only the training split (resp. `False` for test split). By
        default (`None`), loads the entire dataset as-is.
        """
        if self.SPLIT_COLUMN is None and train is not None:
            raise ValueError("Train/test split was specified but dataset has no split column")

        self.root_dir = Path(root)

        if dataset_df is not None:
            self.dataset_csv = None
        else:
            self.dataset_csv = dataset_csv or self.root_dir / self.DEFAULT_CSV_FILENAME
            dataset_df = pd.read_csv(self.dataset_csv)

        columns = [self.SLIDE_ID_COLUMN, self.IMAGE_COLUMN, self.LABEL_COLUMN, self.LABEL_COLUMN,
                   self.SPLIT_COLUMN, self.TILE_X_COLUMN, self.TILE_Y_COLUMN]
        for column in columns:
            if column is not None and column not in dataset_df.columns:
                raise ValueError(f"Expected column '{column}' not found in the dataframe")

        dataset_df = dataset_df.set_index(self.TILE_ID_COLUMN)
        if train is None:
            self.dataset_df = dataset_df
        else:
            split = self.TRAIN_SPLIT_LABEL if train else self.TEST_SPLIT_LABEL
            self.dataset_df = dataset_df[dataset_df[self.SPLIT_COLUMN] == split]

    def __len__(self) -> int:
        return self.dataset_df.shape[0]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        tile_id = self.dataset_df.index[index]
        sample = {
            self.TILE_ID_COLUMN: tile_id,
            **self.dataset_df.loc[tile_id].to_dict()
        }
        sample[self.IMAGE_COLUMN] = str(self.root_dir / sample.pop(self.IMAGE_COLUMN))
        # we're replicating this column because we want to propagate the path to the batch
        sample[self.PATH_COLUMN] = sample[self.IMAGE_COLUMN]
        return sample

    @property
    def slide_ids(self) -> pd.Series:
        return self.dataset_df[self.SLIDE_ID_COLUMN]

    def get_slide_labels(self) -> pd.Series:
        return self.dataset_df.groupby(self.SLIDE_ID_COLUMN)[self.LABEL_COLUMN].agg(pd.Series.mode)

    def get_class_weights(self) -> torch.Tensor:
        slide_labels = self.get_slide_labels()
        classes = np.unique(slide_labels)
        class_weights = compute_class_weight('balanced', classes, slide_labels)
        return torch.as_tensor(class_weights)

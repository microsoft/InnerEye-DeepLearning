#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import pandas as pd

from torchvision.datasets.vision import VisionDataset

from InnerEye.ML.Histopathology.datasets.base_dataset import TilesDataset
from InnerEye.ML.utils.io_util import load_pil_image
from InnerEye.ML.SSL.datamodules_and_datasets.dataset_cls_utils import InnerEyeDataClassBaseWithReturnIndex


class TcgaCrck_TilesDataset(TilesDataset):
    """Dataset class for loading TCGA-CRCk tiles.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str): parent slide ID
    - `'tile_id'` (str)
    - `'image'` (`PIL.Image`): RGB tile
    - `'label'` (str): MSS (0) vs MSIMUT (1)
    """
    TILE_X_COLUMN = TILE_Y_COLUMN = None  # no tile coordinates available
    # This dataset conforms to all other defaults in TilesDataset


class TcgaCrck_TilesDatasetReturnImageLabel(VisionDataset):
    """
    Any dataset used in SSL needs to return a tuple where the first element is the image and the second is a
    class label.
    """

    def __init__(self,
                 root: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 train: Optional[bool] = None,
                 transform: Optional[Callable] = None,
                 **kwargs: Any) -> None:
        super().__init__(root=root, transform=transform)
        self.base_dataset = TcgaCrck_TilesDataset(root=root,
                                                  dataset_csv=dataset_csv,
                                                  dataset_df=dataset_df,
                                                  train=train)

    def __getitem__(self, index: int) -> Tuple:  # type: ignore
        sample = self.base_dataset[index]
        # TODO change to a meaningful evaluation
        image = load_pil_image(sample[self.base_dataset.IMAGE_COLUMN])
        if self.transform:
            image = self.transform(image)
        return image, sample[self.base_dataset.LABEL_COLUMN]

    def __len__(self) -> int:
        return len(self.base_dataset)


class TcgaCrck_TilesDatasetWithReturnIndex(InnerEyeDataClassBaseWithReturnIndex,
                                           TcgaCrck_TilesDatasetReturnImageLabel):
    """
    Any dataset used in SSL needs to inherit from InnerEyeDataClassBaseWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance. Please note that this class needs
    to override __getitem__(), this is why we need a separate PandaTilesDatasetReturnImageLabel.
    """
    @property
    def num_classes(self) -> int:
        return 2

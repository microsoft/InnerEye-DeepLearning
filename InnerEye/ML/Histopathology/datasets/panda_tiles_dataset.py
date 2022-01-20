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


class PandaTilesDataset(TilesDataset):
    """
    Dataset class for loading PANDA tiles.

    Iterating over this dataset returns a dictionary containing:
    - `'slide_id'` (str): parent slide ID (`'image_id'` in the PANDA dataset)
    - `'tile_id'` (str)
    - `'image'` (`PIL.Image`): RGB tile
    - `'mask'` (str): path to mask PNG file
    - `'tile_x'`, `'tile_y'` (int): top-right tile coordinates
    - `'data_provider'`, `'slide_isup_grade'`, `'slide_gleason_score'` (str): parent slide metadata
    """
    LABEL_COLUMN = "slide_isup_grade"
    SPLIT_COLUMN = None  # PANDA does not have an official train/test split
    N_CLASSES = 6

    _RELATIVE_ROOT_FOLDER = Path("PANDA_tiles_20210926-135446/panda_tiles_level1_224")

    def __init__(self,
                 root: Path,
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None) -> None:
        super().__init__(root=Path(root) / self._RELATIVE_ROOT_FOLDER,
                         dataset_csv=dataset_csv,
                         dataset_df=dataset_df,
                         train=None)


class PandaTilesDatasetReturnImageLabel(VisionDataset):
    """
    Any dataset used in SSL needs to return a tuple where the first element is the image and the second is a
    class label.
    """

    def __init__(self,
                 root: Path,
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 transform: Optional[Callable] = None,
                 **kwargs: Any) -> None:
        super().__init__(root=root, transform=transform)
        self.base_dataset = PandaTilesDataset(root=root,
                                              dataset_csv=dataset_csv,
                                              dataset_df=dataset_df)

    def __getitem__(self, index: int) -> Tuple:  # type: ignore
        sample = self.base_dataset[index]
        # TODO change to a meaningful evaluation
        image = load_pil_image(sample[self.base_dataset.IMAGE_COLUMN])
        if self.transform:
            image = self.transform(image)
        return image, 1

    def __len__(self) -> int:
        return len(self.base_dataset)


class PandaTilesDatasetWithReturnIndex(InnerEyeDataClassBaseWithReturnIndex, PandaTilesDatasetReturnImageLabel):
    """
    Any dataset used in SSL needs to inherit from InnerEyeDataClassBaseWithReturnIndex as well as VisionData.
    This class is just a shorthand notation for this double inheritance. Please note that this class needs
    to override __getitem__(), this is why we need a separate PandaTilesDatasetReturnImageLabel.
    """
    @property
    def num_classes(self) -> int:
        return 2

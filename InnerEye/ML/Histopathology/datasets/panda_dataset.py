#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union, Optional

import pandas as pd
from cucim import CuImage
from health_ml.utils import box_utils
from monai.config import KeysCollection
from monai.data.image_reader import ImageReader, WSIReader
from monai.transforms import MapTransform

from InnerEye.ML.Histopathology.datasets.base_dataset import SlidesDataset


class PandaDataset(SlidesDataset):
    """Dataset class for loading files from the PANDA challenge dataset.

    Iterating over this dataset returns a dictionary following the `SlideKey` schema plus meta-data
    from the original dataset (`'data_provider'`, `'isup_grade'`, and `'gleason_score'`).

    Ref.: https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview
    """
    SLIDE_ID_COLUMN = 'image_id'
    IMAGE_COLUMN = 'image'
    MASK_COLUMN = 'mask'
    LABEL_COLUMN = 'isup_grade'

    METADATA_COLUMNS = ('data_provider', 'isup_grade', 'gleason_score')

    DEFAULT_CSV_FILENAME = "train.csv"

    def __init__(self,
                 root: Union[str, Path],
                 dataset_csv: Optional[Union[str, Path]] = None,
                 dataset_df: Optional[pd.DataFrame] = None) -> None:
        super().__init__(root, dataset_csv, dataset_df, validate_columns=False)
        # PANDA CSV does not come with paths for image and mask files
        slide_ids = self.dataset_df[self.SLIDE_ID_COLUMN]
        self.dataset_df[self.IMAGE_COLUMN] = "train_images/" + slide_ids + ".tiff"
        self.dataset_df[self.MASK_COLUMN] = "train_label_masks/" + slide_ids + "_mask.tiff"
        self.validate_columns()


# MONAI's convention is that dictionary transforms have a 'd' suffix in the class name
class ReadImaged(MapTransform):
    """Basic transform to read image files."""
    def __init__(self, reader: ImageReader, keys: KeysCollection,
                 allow_missing_keys: bool = False, **kwargs: Any) -> None:
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.reader = reader
        self.kwargs = kwargs

    def __call__(self, data: Dict) -> Dict:
        for key in self.keys:
            if key in data or not self.allow_missing_keys:
                data[key] = self.reader.read(data[key], **self.kwargs)
        return data


class LoadPandaROId(MapTransform):
    """Transform that loads a pathology slide and mask, cropped to the mask bounding box (ROI).

    Operates on dictionaries, replacing the file paths in `image_key` and `mask_key` with the
    respective loaded arrays, in (C, H, W) format. Also adds the following meta-data entries:
    - `'location'` (tuple): top-right coordinates of the bounding box
    - `'size'` (tuple): width and height of the bounding box
    - `'level'` (int): chosen magnification level
    - `'scale'` (float): corresponding scale, loaded from the file
    """
    def __init__(self, reader: WSIReader, image_key: str = 'image', mask_key: str = 'mask',
                 level: int = 0, margin: int = 0, **kwargs: Any) -> None:
        """
        :param reader: And instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries.
        :param mask_key: Mask key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale files.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        """
        super().__init__([image_key, mask_key], allow_missing_keys=False)
        self.reader = reader
        self.image_key = image_key
        self.mask_key = mask_key
        self.level = level
        self.margin = margin
        self.kwargs = kwargs

    def _get_bounding_box(self, mask_obj: CuImage) -> box_utils.Box:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = mask_obj.resolutions['level_count'] - 1
        scale = mask_obj.resolutions['level_downsamples'][highest_level]
        mask, _ = self.reader.get_data(mask_obj, level=highest_level)  # loaded as RGB PIL image

        foreground_mask = mask[0] > 0  # PANDA segmentation mask is in 'R' channel
        bbox = scale * box_utils.get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox

    def __call__(self, data: Dict) -> Dict:
        mask_obj: CuImage = self.reader.read(data[self.mask_key])
        image_obj: CuImage = self.reader.read(data[self.image_key])

        level0_bbox = self._get_bounding_box(mask_obj)

        # cuCIM/OpenSlide take absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = mask_obj.resolutions['level_downsamples'][self.level]
        scaled_bbox = level0_bbox / scale
        get_data_kwargs = dict(location=(level0_bbox.x, level0_bbox.y),
                               size=(scaled_bbox.w, scaled_bbox.h),
                               level=self.level)
        mask, _ = self.reader.get_data(mask_obj, **get_data_kwargs)  # type: ignore
        data[self.mask_key] = mask[:1]  # PANDA segmentation mask is in 'R' channel
        data[self.image_key], _ = self.reader.get_data(image_obj, **get_data_kwargs)  # type: ignore
        data.update(get_data_kwargs)
        data['scale'] = scale

        mask_obj.close()
        image_obj.close()
        return data

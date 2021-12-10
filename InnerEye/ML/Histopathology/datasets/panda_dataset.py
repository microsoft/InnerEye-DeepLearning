#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Union, Optional

import pandas as pd
from monai.config import KeysCollection
from monai.data.image_reader import ImageReader, WSIReader
from monai.transforms import MapTransform
from openslide import OpenSlide
from torch.utils.data import Dataset

from health_ml.utils import box_utils


class PandaDataset(Dataset):
    """Dataset class for loading files from the PANDA challenge dataset.

    Iterating over this dataset returns a dictionary containing the `'image_id'`, paths to the `'image'`
    and `'mask'` files, and the remaining meta-data from the original dataset (`'data_provider'`,
    `'isup_grade'`, and `'gleason_score'`).

    Ref.: https://www.kaggle.com/c/prostate-cancer-grade-assessment/overview
    """
    def __init__(self, root_dir: Union[str, Path], n_slides: Optional[int] = None,
                 frac_slides: Optional[float] = None) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.train_df = pd.read_csv(self.root_dir / "train.csv", index_col='image_id')
        if n_slides or frac_slides:
            self.train_df = self.train_df.sample(n=n_slides, frac=frac_slides, replace=False,
                                                          random_state=1234)

    def __len__(self) -> int:
        return self.train_df.shape[0]

    def _get_image_path(self, image_id: str) -> Path:
        return self.root_dir / "train_images" / f"{image_id}.tiff"

    def _get_mask_path(self, image_id: str) -> Path:
        return self.root_dir / "train_label_masks" / f"{image_id}_mask.tiff"

    def __getitem__(self, index: int) -> Dict:
        image_id = self.train_df.index[index]
        return {
            'image_id': image_id,
            'image': str(self._get_image_path(image_id).absolute()),
            'mask': str(self._get_mask_path(image_id).absolute()),
            **self.train_df.loc[image_id].to_dict()
        }


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

    def _get_bounding_box(self, mask_obj: OpenSlide) -> box_utils.Box:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = mask_obj.level_count - 1
        scale = mask_obj.level_downsamples[highest_level]
        mask, _ = self.reader.get_data(mask_obj, level=highest_level)  # loaded as RGB PIL image

        foreground_mask = mask[0] > 0  # PANDA segmentation mask is in 'R' channel
        bbox = scale * box_utils.get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox

    def __call__(self, data: Dict) -> Dict:
        mask_obj: OpenSlide = self.reader.read(data[self.mask_key])
        image_obj: OpenSlide = self.reader.read(data[self.image_key])

        level0_bbox = self._get_bounding_box(mask_obj)

        # OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        scale = mask_obj.level_downsamples[self.level]
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

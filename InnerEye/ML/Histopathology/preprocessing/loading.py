import logging
from typing import Dict, Optional, Tuple

import numpy as np
import skimage.filters
from health_ml.utils import box_utils
from monai.data.image_reader import WSIReader
from monai.transforms import MapTransform

from InnerEye.ML.Histopathology.utils.naming import SlideKey

try:
    from cucim import CuImage
except:
    logging.warning("cucim library not available, code may fail.")


def get_luminance(slide: np.ndarray) -> np.ndarray:
    """Compute a grayscale version of the input slide.

    :param slide: The RGB image array in (*, C, H, W) format.
    :return: The single-channel luminance array as (*, H, W).
    """
    # TODO: Consider more sophisticated luminance calculation if necessary
    return slide.mean(axis=-3)  # type: ignore


def segment_foreground(slide: np.ndarray, threshold: Optional[float] = None) \
        -> Tuple[np.ndarray, float]:
    """Segment the given slide by thresholding its luminance.

    :param slide: The RGB image array in (*, C, H, W) format.
    :param threshold: Pixels with luminance below this value will be considered foreground.
    If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
    :return: A tuple containing the boolean output array in (*, H, W) format and the threshold used.
    """
    luminance = get_luminance(slide)
    if threshold is None:
        threshold = skimage.filters.threshold_otsu(luminance)
    return luminance < threshold, threshold


def load_slide_at_level(reader: WSIReader, slide_obj: 'CuImage', level: int) -> np.ndarray:
    """Load full slide array at the given magnification level.

    This is a manual workaround for a MONAI bug (https://github.com/Project-MONAI/MONAI/issues/3415)
    fixed in a currently unreleased PR (https://github.com/Project-MONAI/MONAI/pull/3417).

    :param reader: A MONAI `WSIReader` using cuCIM backend.
    :param slide_obj: The cuCIM image object returned by `reader.read(<image_file>)`.
    :param level: Index of the desired magnification level as defined in the `slide_obj` headers.
    :return: The loaded image array in (C, H, W) format.
    """
    size = slide_obj.resolutions['level_dimensions'][level][::-1]
    slide, _ = reader.get_data(slide_obj, size=size, level=level)  # loaded as RGB PIL image
    return slide


class LoadROId(MapTransform):
    """Transform that loads a pathology slide, cropped to an estimated bounding box (ROI).

    Operates on dictionaries, replacing the file path in `image_key` with the loaded array in
    (C, H, W) format. Also adds the following entries:
    - `SlideKey.ORIGIN` (tuple): top-right coordinates of the bounding box
    - `SlideKey.SCALE` (float): corresponding scale, loaded from the file
    - `SlideKey.FOREGROUND_THRESHOLD` (float): threshold used to segment the foreground
    """

    def __init__(self, reader: WSIReader, image_key: str = SlideKey.IMAGE, level: int = 0,
                 margin: int = 0, foreground_threshold: Optional[float] = None) -> None:
        """
        :param reader: And instance of MONAI's `WSIReader`.
        :param image_key: Image key in the input and output dictionaries.
        :param level: Magnification level to load from the raw multi-scale file.
        :param margin: Amount in pixels by which to enlarge the estimated bounding box for cropping.
        :param foreground_threshold: Pixels with luminance below this value will be considered foreground.
        If `None` (default), an optimal threshold will be estimated automatically using Otsu's method.
        """
        super().__init__([image_key], allow_missing_keys=False)
        self.reader = reader
        self.image_key = image_key
        self.level = level
        self.margin = margin
        self.foreground_threshold = foreground_threshold

    def _get_bounding_box(self, slide_obj: 'CuImage') -> Tuple[box_utils.Box, float]:
        # Estimate bounding box at the lowest resolution (i.e. highest level)
        highest_level = slide_obj.resolutions['level_count'] - 1
        scale = slide_obj.resolutions['level_downsamples'][highest_level]
        slide = load_slide_at_level(self.reader, slide_obj, level=highest_level)

        foreground_mask, threshold = segment_foreground(slide, self.foreground_threshold)
        bbox = scale * box_utils.get_bounding_box(foreground_mask).add_margin(self.margin)
        return bbox, threshold

    def __call__(self, data: Dict) -> Dict:
        image_obj: CuImage = self.reader.read(data[self.image_key])

        level0_bbox, threshold = self._get_bounding_box(image_obj)

        # cuCIM/OpenSlide takes absolute location coordinates in the level 0 reference frame,
        # but relative region size in pixels at the chosen level
        origin = (level0_bbox.x, level0_bbox.y)
        scale = image_obj.resolutions['level_downsamples'][self.level]
        scaled_bbox = level0_bbox / scale

        data[self.image_key], _ = self.reader.get_data(image_obj, location=origin, level=self.level,
                                                       size=(scaled_bbox.w, scaled_bbox.h))
        data[SlideKey.ORIGIN] = origin
        data[SlideKey.SCALE] = scale
        data[SlideKey.FOREGROUND_THRESHOLD] = threshold

        image_obj.close()
        return data

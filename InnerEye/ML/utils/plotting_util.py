#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Tuple

import numpy as np
from scipy.ndimage import find_objects

from InnerEye.ML.utils.surface_distance_utils import Plane


def get_view_dim_and_origin(plane: Plane) -> Tuple[int, str]:
    """
    Get the axis along which to slice, as well as the orientation of the origin, to ensure images
    are plotted as expected
    :param plane: the plane in which to plot (i.e. axial, sagittal or coronal)
    :return:
    """
    # default origin is sagittal
    view_dim = 1
    origin = 'lower'
    plane_name = plane.value
    if plane_name == 'CORONAL':
        view_dim = 0
    elif plane_name == 'AXIAL':
        view_dim = 2
        origin = 'upper'
    return view_dim, origin


def get_cropped_axes(image: np.ndarray, boundary_width: int = 5) -> Tuple[slice, ...]:
    """
    Return the min and max values on both x and y axes where the image is not empty
    Method: find the min and max of all non-zero pixels in the image, and add a border
    :param image: the image to be cropped
    :param boundary_width: number of pixels boundary to add around bounding box
    :return:
    """
    x_lim = image.shape[0]
    y_lim = image.shape[1]

    # noinspection PyUnresolvedReferences
    slice_x, slice_y = find_objects(image > 0)[0]
    new_slice_x_min = max(0, slice_x.start - boundary_width)
    new_slice_x_max = min(x_lim, slice_x.stop + boundary_width)
    new_slice_x = slice(new_slice_x_min, new_slice_x_max)

    new_slice_y_min = max(0, slice_y.start - boundary_width)
    new_slice_y_max = min(y_lim, slice_y.stop + boundary_width)
    new_slice_y = slice(new_slice_y_min, new_slice_y_max)

    return tuple([new_slice_x, new_slice_y])

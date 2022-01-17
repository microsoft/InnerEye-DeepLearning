#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List
import numpy as np


def location_selected_tiles(tile_coords: np.ndarray,
                            location_bbox: List[int],
                            level: int) -> np.ndarray:
    """ Return the scaled and shifted tile co-ordinates for selected tiles in the slide.
    :param tile_coords: XY tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]) in original resolution.
    :param location_bbox: Location of the bounding box on the slide in original resolution.
    :param level: The downsampling level (e.g. 0, 1, 2) of the tiles if available. 
    (e.g. PANDA levels are 0 for original, 1 for 4x downsampled, 2 for 16x downsampled).
    """
    level_dict = {0: 1, 1: 4, 2: 16}
    factor = level_dict[level]

    x_tr, y_tr = location_bbox
    tile_xs, tile_ys = tile_coords.T
    tile_xs = tile_xs - x_tr 
    tile_ys = tile_ys - y_tr 
    tile_xs = tile_xs//factor
    tile_ys = tile_ys//factor

    sel_coords = np.transpose([tile_xs.tolist(), tile_ys.tolist()])

    return sel_coords

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.axes import Axes
from matplotlib.image import AxesImage
import matplotlib.patches as patches 
import matplotlib.collections as collection


def plot_heatmap_selected_tiles(tile_coords: np.array,
                                tile_values: np.ndarray,
                                location_bbox: List[int],
                                tile_size: int,
                                level: int,
                                ax: Optional[Axes] = None) -> AxesImage:
    """Plots a 2D heatmap for selected tiles to overlay on the slide.
    :param tile_coords: XY tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]).
    :param tile_values: Scalar values of the tiles (shape: [N]).
    :param location_bbox: Location of the bounding box of the slide.
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original, 1 for 4x downsampled, 2 for 16x downsampled).
    :param tile_size: Size of each tile.
    :param ax: Axes onto which to plot the heatmap (default: current axes).
    """
    if ax is None:
        ax = plt.gca()

    level_dict = {"0": 1, "1": 4, "2": 16}
    factor = level_dict[str(level)]
    x_tr, y_tr = location_bbox
    tile_xs, tile_ys = tile_coords.T
    tile_xs = tile_xs - x_tr 
    tile_ys = tile_ys - y_tr 
    tile_xs = tile_xs//factor
    tile_ys = tile_ys//factor

    cmap = plt.cm.get_cmap('jet')
    sel_coords = np.transpose([tile_xs.tolist(), tile_ys.tolist()])
    rects = []
    for i in range(sel_coords.shape[0]):
        rect = patches.Rectangle((sel_coords[i][0], sel_coords[i][1]), tile_size, tile_size)
        rects.append(rect)

    pc = collection.PatchCollection(rects, match_original=True, cmap=cmap, alpha=.5, edgecolor=None)
    pc.set_array(np.array(tile_values))
    pc.set_clim([0, 1])
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax)
    return ax

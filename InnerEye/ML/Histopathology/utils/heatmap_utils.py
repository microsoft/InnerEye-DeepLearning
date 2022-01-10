import io
from typing import Any, Optional, Sequence, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
import matplotlib.patches as patches 
import matplotlib.collections as collection


def assemble_heatmap(tile_coords: np.ndarray, tile_values: np.ndarray, tile_size: int, level: int,
                     fill_value: float = np.nan, pad: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Assembles a 2D heatmap from sequences of tile coordinates and values.

    :param tile_coords: XY tile coordinates, assumed to be spaced by multiples of `tile_size` (shape: [N, 2]).
    :param tile_values: Scalar values of the tiles (shape: [N]).
    :param tile_size: Size of each tile; must be >0.
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original, 1 for 4x downsampled, 2 for 16x downsampled).
    :param fill_value: Value to assign to empty elements (default: `NaN`).
    :param pad: If positive, pad the heatmap by `pad` elements on all sides (default: no padding).
    :return: A tuple containing:
        - `heatmap`: The 2D heatmap with the smallest dimensions to contain all given tiles, with
          optional padding.
        - `origin`: The lowest XY coordinates in the space of `tile_coords`. If `pad > 0`, this is
          offset to match the padded margin.
    """
    if tile_coords.shape[0] != tile_values.shape[0]:
        raise ValueError(f"Tile coordinates and values must have the same length, "
                         f"got {tile_coords.shape[0]} and {tile_values.shape[0]}")

    level_dict = {"0": 1, "1": 4, "2": 16}
    factor = level_dict[str(level)]
    tile_coords_scaled = tile_coords//factor
    tile_xs, tile_ys = tile_coords_scaled.T

    tile_xs = tile_xs - tile_size          # top-left corner from top-right corner
    x_min, x_max = min(tile_xs), max(tile_xs)
    y_min, y_max = min(tile_ys), max(tile_ys)

    n_tiles_x = (x_max - x_min) // tile_size + 1
    n_tiles_y = (y_max - y_min) // tile_size + 1
    heatmap = np.full((n_tiles_y, n_tiles_x), fill_value)

    tile_js = (tile_xs - x_min) // tile_size
    tile_is = (tile_ys - y_min) // tile_size
    heatmap[tile_is, tile_js] = tile_values
    origin = np.array([x_min, y_min])

    if pad > 0:
        heatmap = np.pad(heatmap, pad, mode='constant', constant_values=fill_value)
        origin -= tile_size * pad  # offset the origin to match the padded margin

    return heatmap, origin


def plot_heatmap(heatmap: np.ndarray, tile_size: int, origin: Sequence[int], ax: Optional[Axes] = None, **imshow_kwargs: Any) -> AxesImage:
    """Plot a 2D heatmap to overlay on the slide.

    :param heatmap: The 2D scalar heatmap.
    :param tile_size: Size of each tile.
    :param origin: XY coordinates of the heatmap's top-left corner.
    :param ax: Axes onto which to plot the heatmap (default: current axes).
    :param imshow_kwargs: Kwargs for `plt.imshow()` (e.g. `alpha`, `cmap`, `interpolation`).
    :return: The output of `plt.imshow()` to allow e.g. plotting a colorbar.
    """
    if ax is None:
        ax = plt.gca()
    heatmap_width = tile_size * heatmap.shape[1]
    heatmap_height = tile_size * heatmap.shape[0]
    offset = tile_size * 0.5
    extent = (
        origin[0] - offset,  # left
        origin[0] + heatmap_width - offset,  # right
        origin[1] + heatmap_height - offset,  # bottom
        origin[1] - offset  # top
    )
    h = ax.imshow(heatmap, extent=extent, **imshow_kwargs)
    cb = plt.colorbar(h, ax=ax)
    cb.set_alpha(1)
    cb.draw_all()
    return ax


def figure_to_image(fig: Optional[Figure] = None, **savefig_kwargs: Any) -> PIL.Image:
    """Converts a Matplotlib figure into an image for logging and display.

    :param fig: Input Matplotlib figure object (default: current figure).
    :param savefig_kwargs: Kwargs for `fig.savefig()` (e.g. `dpi`, `bbox_inches`).
    :return: Rasterised PIL image in RGBA format.
    """
    if fig is None:
        fig = plt.gcf()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', **savefig_kwargs)
    buffer.seek(0)
    image = PIL.Image.open(buffer).convert('RGBA')
    buffer.close()
    return image


def plot_slide_from_tiles(tiles: np.array,
                          tile_coords: np.array,
                          level: int, tile_size: int,
                          width: int, height: int,
                          ax: Optional[Axes] = None,
                          **imshow_kwargs: Any) -> AxesImage:
    """Reconstructs a slide given the tiles at a certain magnification.

    :param tiles: Tiles of the slide (shape: [N, H, W, C]).
    :param tile_coords: Top-right coordinates of tiles  (shape: [N, 2]).
    :param level: Magnification at which tiles are available (e.g. PANDA levels are 0 for original, 1 for 4x downsampled, 2 for 16x downsampled).
    :param tile_size: Size of each tile.
    :param width: Width of slide.
    :param height: Height of slide.
    :param ax: Axes onto which to plot the heatmap (default: current axes).
    """
    level_dict = {"0": 1, "1": 4, "2": 16}
    factor = level_dict[str(level)]
    tile_coords_scaled = tile_coords//factor
    xs, ys = tile_coords_scaled.T
    x_min = min(xs) - tile_size  # top-left corner from top-right corner
    y_min = min(ys)
    offset = tile_size * 0.5
    if ax is None:
        ax = plt.gca()
    for i in range(tiles.shape[0]):
        x, y = tile_coords_scaled[i]
        x = x - tile_size        # top-left corner from top-right corner
        x = x - x_min
        y = y - y_min
        ax.imshow(tiles[i], extent=(x-offset, x+tile_size-offset, y+tile_size-offset, y-offset), **imshow_kwargs)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    return ax


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
    x_min = x_tr//factor
    y_min = y_tr//factor

    tile_coords_scaled = tile_coords//factor
    tile_xs, tile_ys = tile_coords_scaled.T
    tile_xs = tile_xs - x_min 
    tile_ys = tile_ys - y_min 
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

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.pyplot import Axes

from InnerEye.Common.type_annotations import PathOrString, TupleFloat2, TupleFloat3
from InnerEye.ML.dataset.full_image_dataset import Sample
from InnerEye.ML.photometric_normalization import PhotometricNormalization
from InnerEye.ML.utils import plotting_util
from InnerEye.ML.utils.image_util import binaries_from_multi_label_array, get_largest_z_slice
from InnerEye.ML.utils.ml_util import check_size_matches
from InnerEye.ML.utils.surface_distance_utils import Plane, extract_border

VAL_DICE_PREFIX = "Val_Dice/"
BACKGROUND_0 = "background_0"


def is_val_dice(name: str) -> bool:
    """
    Returns true if the given metric name is a Dice score on the validation set,
    for a class that is not the background class.
    :param name:
    :return:
    """
    return name.startswith(VAL_DICE_PREFIX) and BACKGROUND_0 not in name


def get_val_dice_names(metric_names: Iterable[str]) -> List[str]:
    """
    Returns a list of those metric names from the argument that fulfill the is_val_dice predicate.
    :param metric_names:
    :return:
    """
    return [name for name in metric_names if is_val_dice(name)]


def plot_loss_per_epoch(metrics: Dict[str, Any], metric_name: str, label: Optional[str] = None) -> int:
    """
    Adds a plot of loss (y-axis) versus epoch (x-axis) to the current plot, if the metric
    is present in the metrics dictionary.
    :param metrics: A dictionary of metrics.
    :param metric_name: The name of the single metric to plot.
    :param label: The label for the series that will be plotted.
    :return: 1 if the metric is present in the dictionary and was plotted, 0 otherwise.
    """
    if label is None:
        label = metric_name
    m = metrics.get(metric_name, None)
    if isinstance(m, dict) and "epoch" in m and "loss" in m:
        plt.plot(m["epoch"], m["loss"], label=label)
        return 1
    return 0


def plot_val_dice_per_epoch(metrics: Dict[str, Any]) -> int:
    """
    Creates a plot of all validation Dice scores per epoch, for all classes apart from background.
    :param metrics:
    :return: The number of series that were plotted in the graph. Can return 0 if the metrics dictionary
    does not contain any validation Dice score.
    """
    plt.clf()
    series_count = 0
    for metric_name in get_val_dice_names(metrics.keys()):
        metric_without_prefix = metric_name[len(VAL_DICE_PREFIX):]
        series_count += plot_loss_per_epoch(metrics, metric_name, label=metric_without_prefix)
    if series_count > 0:
        plt.ylim(0, 1)
        plt.xlabel("Epoch")
        plt.ylabel("Val_Dice")
    return series_count


def add_legend(series_count: int) -> None:
    """
    Adds a legend to the present plot, with the column layout depending on the number of series.
    :param series_count:
    :return:
    """
    num_columns = 2 if series_count > 8 else 1
    plt.legend(ncol=num_columns, loc="upper left", fontsize="x-small")


def resize_and_save(width_inch: int, height_inch: int, filename: PathOrString, dpi: int = 150) -> None:
    """
    Resizes the present figure to the given (width, height) in inches, and saves it to the given filename.
    :param width_inch: The width of the figure in inches.
    :param height_inch: The height of the figure in inches.
    :param filename: The filename to save to.
    :param dpi: Image resolution in dots per inch
    """
    fig = plt.gcf()
    fig.set_size_inches(width_inch, height_inch)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.1)


def plot_image_and_label_contour(image: np.ndarray,
                                 labels: Union[np.ndarray, List[np.ndarray]],
                                 plot_file_name: Path,
                                 contour_arguments: Optional[Union[Dict, List[Dict]]] = None,
                                 image_range: Optional[TupleFloat2] = None) -> Path:
    """
    Creates a plot that shows the given 2D image in greyscale, and overlays a contour that shows
    where the 'labels' array has value 1.
    :param image: A 2D image
    :param labels: A binary 2D image, or a list of binary 2D images. A contour will be plotted for each of those
    binary images.
    :param contour_arguments: A dictionary of keyword arguments that will be passed directly into matplotlib's
    contour function. Can also be a list of dictionaries, with one dict per entry in the 'labels' argument.
    :param image_range: If provided, the image will be plotted using the given range for the color limits.
    If None, the minimum and maximum image values will be mapped to the endpoints of the color map.
    :param plot_file_name: The file name that should be used to save the plot.
    """
    if image.ndim != 2:
        raise ValueError("The 'image' parameter should be a 2D array, but got shape {}".format(image.shape))
    if contour_arguments is None:
        contour_arguments = {'colors': 'r'}
    plt.clf()
    norm = None if image_range is None else plt.Normalize(vmin=image_range[0], vmax=image_range[1])
    plt.imshow(image, norm=norm, cmap="Greys_r")
    plt.xticks([])
    plt.yticks([])

    def plot_contour(contour_data: np.ndarray, extra_args: Dict) -> None:
        if contour_data.shape != image.shape:
            raise ValueError("The image and the contour data should have matching size, but got {} and {}"
                             .format(image.shape, contour_data.shape))
        plt.contour(contour_data, levels=[.5], **extra_args)

    if isinstance(labels, np.ndarray) and isinstance(contour_arguments, dict):
        plot_contour(labels, contour_arguments)
    elif isinstance(labels, list) and isinstance(contour_arguments, list):
        for label, contour in zip(labels, contour_arguments):
            plot_contour(label, contour)
    else:
        raise ValueError("Combination of input arguments is not recognized.")

    result = plot_file_name.with_suffix(".png")
    resize_and_save(width_inch=5, height_inch=5, filename=result)
    return result


def _plot_single_image_stats(image: np.ndarray,
                             mask: np.ndarray,
                             z_slice: int,
                             image_axes: Axes,
                             hist_axes: Axes,
                             box_axes: Axes) -> None:
    data = image.flatten() if mask is None else image[mask > 0].flatten()
    box_axes.boxplot(data, notch=False, vert=False, sym=".", whis=[5, 95])
    hist_axes.hist(data, bins=30)
    image_axes.imshow(image[z_slice, :, :], cmap="Greys_r")
    image_axes.set_xticks([])
    image_axes.set_yticks([])
    # The histogram limits represent the full data range, set that also for the box plot
    # (box plot may show smaller range if no outliers are plotted)
    xlims = hist_axes.get_xlim()
    box_axes.set_xlim(left=xlims[0], right=xlims[1])
    box_axes.set_xticks([])  # Ticks match those of histogram anyway
    box_axes.set_yticks([])  # don't need that 1 tick mark
    hist_axes.set_yticks([])  # Number of voxels is not relevant


def plot_before_after_statistics(image_before: np.ndarray,
                                 image_after: np.ndarray,
                                 mask: Optional[np.ndarray],
                                 z_slice: int,
                                 normalizer_status_message: Optional[str],
                                 plot_file_name: Path) -> Path:
    """
    Creates a plot in a PNG file that describes the pixel value distribution of two 3D images in Z x Y x X order,
    that were obtained before and after a transformation of pixel values.
    The plot contains histograms, box plots, and visualizations of a single XY slice at z_slice.
    If a mask argument is provided, only the image pixel values inside of the mask will be plotted.
    :param image_before: The first image for which to plot statistics.
    :param image_after: The second image for which to plot statistics.
    :param mask: Indicators with 1 for foreground, 0 for background. If None, plot statistics for all image pixels.
    :param z_slice: The z position for visualizing a single example slice.
    :param normalizer_status_message: A string with diagnostic information that will be set as the plot title.
    :param plot_file_name: The file name under which to save the plot. A PNG extension will be added.
    :return: The path of the created PNG file.
    """
    plt.clf()
    fig = plt.gcf()
    # Grid specification as left, bottom, width, height
    image_width = 0.2
    image_height = 0.4
    image_bottom = 0.2
    box_width = 0.3
    box_height = 0.2
    box_bottom = 0.1
    hist_height = 0.6
    hist_bottom = box_bottom + box_height
    left = 0.0
    image1 = fig.add_axes([left, image_bottom, image_width, image_bottom + image_height])
    left += image_width
    box1 = fig.add_axes([left, box_bottom, box_width, box_height])
    hist1 = fig.add_axes([left, hist_bottom, box_width, hist_height])
    left += box_width
    box2 = fig.add_axes([left, box_bottom, box_width, box_height])
    hist2 = fig.add_axes([left, hist_bottom, box_width, hist_height])
    left += box_width
    image2 = fig.add_axes([left, image_bottom, image_width, image_bottom + image_height])
    _plot_single_image_stats(image_before, mask, z_slice, image1, hist1, box1)
    _plot_single_image_stats(image_after, mask, z_slice, image2, hist2, box2)
    plt.sca(hist1)
    plt.title(f"{plot_file_name.name}: size {image_before.shape}", fontsize=14)
    plt.sca(hist2)
    plt.title(normalizer_status_message, fontsize=14)
    result = plot_file_name.with_suffix(".png")
    resize_and_save(width_inch=10, height_inch=5, filename=result)
    return result


def plot_normalization_result(loaded_images: Sample,
                              normalizer: PhotometricNormalization,
                              result_folder: Path,
                              result_prefix: str = "",
                              image_range: Optional[TupleFloat2] = None,
                              channel_index: int = 0,
                              class_index: int = 1,
                              contour_file_suffix: str = "") -> List[Path]:
    """
    Creates two PNG plots that summarize the result of photometric normalization of the first channel in the
    sample image.
    The first plot contains pixel value histograms before and after photometric normalization.
    The second plot contains the normalized image, overlayed with contours for the foreground pixels,
    at the slice where the foreground has most pixels.
    :param loaded_images: An instance of Sample with the image and the labels. The first channel of the image will
    be plotted.
    :param image_range: The image value range that will be mapped to the color map. If None, the full image range
    will be mapped to the colormap.
    :param normalizer: The photometric normalization that should be applied.
    :param result_folder: The folder into which the resulting PNG files should be written.
    :param result_prefix: The prefix for all output filenames.
    :param channel_index: Compute normalization results for this channel.
    :param class_index: When plotting image/contour overlays, use this class.
    :param contour_file_suffix: Use this suffix for the file name that contains the image/contour overlay.
    :return: The paths of the two PNG files that the function writes.
    """
    # Labels are encoded with background and a single foreground class. We need the
    # slice with largest number of foreground voxels
    ground_truth = loaded_images.labels[class_index, ...]
    largest_gt_slice = get_largest_z_slice(ground_truth)
    first_channel = loaded_images.image[channel_index, ...]
    filename_stem = f"{result_prefix}{loaded_images.patient_id:03d}_slice_{largest_gt_slice:03d}"
    normalized_image = normalizer.transform(loaded_images.image, loaded_images.mask)[channel_index, ...]

    before_after_plot = \
        plot_before_after_statistics(first_channel,
                                     normalized_image,
                                     loaded_images.mask,
                                     z_slice=largest_gt_slice,
                                     normalizer_status_message=normalizer.status_of_most_recent_call,
                                     plot_file_name=result_folder / filename_stem)
    image_contour_plot = \
        plot_image_and_label_contour(image=normalized_image[largest_gt_slice, ...],
                                     labels=ground_truth[largest_gt_slice, ...],
                                     contour_arguments={'colors': 'r'},
                                     image_range=image_range,
                                     plot_file_name=result_folder / f"{filename_stem}_contour{contour_file_suffix}")
    return [before_after_plot, image_contour_plot]


def plot_contours_for_all_classes(sample: Sample,
                                  segmentation: np.ndarray,
                                  foreground_class_names: List[str],
                                  result_folder: Path,
                                  result_prefix: str = "",
                                  image_range: Optional[TupleFloat2] = None,
                                  channel_index: int = 0) -> List[Path]:
    """
    Creates a plot with the image, the ground truth, and the predicted segmentation overlaid. One plot is created
    for each class, each plotting the Z slice where the ground truth has most pixels.
    :param sample: The image sample, with the photonormalized image and the ground truth labels.
    :param segmentation: The predicted segmentation: multi-value, size Z x Y x X.
    :param foreground_class_names: The names of all classes, excluding the background class.
    :param result_folder: The folder into which the resulting plot PNG files should be written.
    :param result_prefix: A string prefix that will be used for all plots.
    :param image_range: The minimum and maximum image values that will be mapped to the color map ranges.
    If None, use the actual min and max values.
    :param channel_index: The index of the image channel that should be plotted.
    :return: The paths to all generated PNG files.
    """
    check_size_matches(sample.labels[0], segmentation)
    num_classes = sample.labels.shape[0]
    if len(foreground_class_names) != num_classes - 1:
        raise ValueError(
            f"Labels tensor indicates {num_classes} classes, but got {len(foreground_class_names)} foreground "
            f"class names: {foreground_class_names}")
    plot_names: List[Path] = []
    image = sample.image[channel_index, ...]
    contour_arguments = [{'colors': 'r'}, {'colors': 'b', 'linestyles': 'dashed'}]
    binaries = binaries_from_multi_label_array(segmentation, num_classes)
    for class_index, binary in enumerate(binaries):
        if class_index == 0:
            continue
        ground_truth = sample.labels[class_index, ...]
        largest_gt_slice = get_largest_z_slice(ground_truth)
        labels_at_largest_gt = ground_truth[largest_gt_slice]
        segmentation_at_largest_gt = binary[largest_gt_slice, ...]
        class_name = foreground_class_names[class_index - 1]
        patient_id = sample.patient_id
        if isinstance(patient_id, str):
            patient_id_str = patient_id
        else:
            patient_id_str = f"{patient_id:03d}"
        filename_stem = f"{result_prefix}{patient_id_str}_{class_name}_slice_{largest_gt_slice:03d}"
        plot_file = plot_image_and_label_contour(image=image[largest_gt_slice, ...],
                                                 labels=[labels_at_largest_gt, segmentation_at_largest_gt],
                                                 contour_arguments=contour_arguments,
                                                 image_range=image_range,
                                                 plot_file_name=result_folder / filename_stem)

        plot_names.append(plot_file)
    return plot_names


def segmentation_and_groundtruth_plot(prediction: np.ndarray, ground_truth: np.ndarray, subject_id: int,
                                      structure_name: str, plane: Plane, output_img_dir: Path, annotator: str = None,
                                      save_fig: bool = True) -> None:
    """
    Plot predicted and the ground truth segmentations. Always plots the middle slice (to match surface distance
    plots), which can sometimes lead to an empty plot.
    :param prediction: 3D volume (X x Y x Z) of predicted segmentation
    :param ground_truth: 3D volume (X x Y x Z) of ground truth segmentation
    :param subject_id: ID of subject for annotating plot
    :param structure_name: Name of structure for annotating plot
    :param plane: The plane to view images in  (axial, sagittal or coronal)
    :param output_img_dir: The dir in which to store the plots
    :param annotator: Optional annotator name for annotating plot
    :param save_fig: If True, saves image. Otherwise displays it.
    :return:
    """
    prediction_contour = extract_border(prediction, connectivity=3)
    gt_contour = extract_border(ground_truth, connectivity=3)

    fig, ax = plt.subplots()

    view_dim, origin = plotting_util.get_view_dim_and_origin(plane)
    midpoint = prediction_contour.shape[view_dim] // 2
    pred_slice = np.take(prediction_contour, midpoint, axis=view_dim)
    gt_slice = np.take(gt_contour, midpoint, axis=view_dim)
    total_pixels = pred_slice + gt_slice

    try:
        bounding_box = plotting_util.get_cropped_axes(total_pixels)
    except IndexError:
        bounding_box = tuple([slice(0, total_pixels.shape[0] + 1), slice(0, total_pixels.shape[1] + 1)])

    ax.imshow(gt_slice[bounding_box], cmap="Greens", origin=origin, aspect='equal')
    ax.imshow(pred_slice[bounding_box], cmap="Reds", origin=origin, alpha=0.7, aspect='equal')

    if annotator:
        annot_str = annotator
    else:
        annot_str = ""

    ax.set_aspect('equal')
    fig.suptitle(f"{subject_id} {structure_name} ground truth and pred - {annot_str}")

    if save_fig:
        figpath = str(Path("outputs/") / output_img_dir
                      / f"{subject_id}_{structure_name}_ground_truth_and_pred_{annot_str}.png")
        print(f"saving to {figpath}")
        resize_and_save(5, 5, figpath)
    else:
        fig.show()


def surface_distance_ground_truth_plot(ct: np.ndarray, ground_truth: np.ndarray, sds_full: np.ndarray, subject_id: int,
                                       structure: str, plane: Plane, output_img_dir: Path, dice: float = None,
                                       save_fig: bool = True,
                                       annotator: str = None) -> None:
    """
    Plot surface distances where prediction > 0, with ground truth contour
    :param ct: CT scan
    :param ground_truth: Ground truth segmentation
    :param sds_full: Surface distances (full= where prediction > 0)
    :param subject_id: ID of subject for annotating plot
    :param structure: Name of structure for annotating plot
    :param plane: The plane to view images in  (axial, sagittal or coronal)
    :param output_img_dir: The dir in which to store the plots
    :param dice: Optional dice score for annotating plot
    :param save_fig: If True, saves image. Otherwise displays it.
    :param annotator: Optional annotator name for annotating plot
    :return:
    """
    # get dimension to slice across to get the best 2D view
    view_dim, origin = plotting_util.get_view_dim_and_origin(plane)
    midpoint = ground_truth.shape[view_dim] // 2

    # Take image slices and mask where necessary
    sds_full_slice = np.take(sds_full, midpoint, axis=view_dim)
    total_pixels = sds_full_slice
    # If surface distance array covers everywhere with pred > 0, mask at some threshold else centre of every
    # structure will be red
    masked_sds_full_slice = np.ma.masked_where(sds_full_slice == 0, sds_full_slice)

    gt_contour = extract_border(ground_truth, connectivity=3)
    gt_contour_slice = np.take(gt_contour, midpoint, axis=view_dim)

    total_pixels += gt_contour_slice.astype(float)

    try:
        bounding_box = plotting_util.get_cropped_axes(total_pixels)
    except IndexError:
        bounding_box = tuple([slice(0, total_pixels.shape[0] + 1), slice(0, total_pixels.shape[1] + 1)])

    fig, ax = plt.subplots()
    black_cmap = colors.ListedColormap('black')
    sds_cmap = plt.get_cmap("RdYlGn_r")

    bounds = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    sds_norm = colors.BoundaryNorm(bounds, sds_cmap.N)

    # plot pixels outside of border in black
    masked_external_pixels = np.ma.masked_where(ground_truth == 1, ground_truth)
    masked_external_slice = np.take(masked_external_pixels, midpoint, axis=view_dim)

    if ct is not None:
        ct_slice = np.take(ct, midpoint, axis=view_dim)
        ax.imshow(ct_slice[bounding_box], cmap="Greys", origin=origin)
        ax.imshow(masked_external_slice[bounding_box], cmap=black_cmap, origin=origin, alpha=0.7)
    else:
        gt_slice = np.take(ground_truth, midpoint, axis=view_dim)
        ax.imshow(gt_slice[bounding_box], cmap='Greys_r', origin=origin)

    cb = ax.imshow(masked_sds_full_slice[bounding_box], cmap=sds_cmap, norm=sds_norm, origin=origin, alpha=0.7)

    fig.colorbar(cb)

    # Plot title
    dice_str = str(dice) if dice else ""
    annot_str = annotator if annotator else ""
    fig.suptitle(f'{subject_id} {structure} sds - {annot_str}. Dice: {dice_str}')

    # Resize image
    ax.set_aspect('equal')

    if save_fig:
        figpath = Path("outputs") / output_img_dir / f"{int(subject_id):03d}_{structure}_sds2_{annot_str}.png"
        print(f"saving to {str(figpath)}")
        resize_and_save(5, 5, figpath)
    else:
        fig.show()


def scan_with_transparent_overlay(scan: np.ndarray,
                                  overlay: np.ndarray,
                                  dimension: int,
                                  position: int,
                                  spacing: TupleFloat3) -> None:
    """
    Creates a plot with one slice of a (CT) scan, with a transparent overlay that contains a second piece of
    information in the range [0, 1]. High values of the `overlay` are shown as opaque red, low values as transparent
    red.
    Plots are created in the current axis.
    :param scan: A 3-dimensional image in (Z, Y, X) ordering
    :param overlay: A 3-dimensional image in (Z, Y, X) ordering, with values between 0 and 1.
    :param dimension: The array dimension along with the plot should be created. dimension=0 will generate
    an axial slice.
    :param position: The index in the chosen dimension where the plot should be created.
    :param spacing: The tuple of voxel spacings, in (Z, Y, X) order.
    """
    if dimension < 0 or dimension > 2:
        raise ValueError(f"Dimension must be in the range [0, 2], but got: {dimension}")
    if position < 0 or position >= scan.shape[dimension]:
        raise IndexError(f"Position is outside valid range: {position}")
    slicers = []
    for i in range(0, 3):
        if i == dimension:
            slicers.append(slice(position, position + 1))
        else:
            slicers.append(slice(0, scan.shape[i]))
    # Slice both the scan and the overlay
    scan_sliced = scan[slicers[0], slicers[1], slicers[2]].squeeze(axis=dimension)
    overlay_sliced = overlay[slicers[0], slicers[1], slicers[2]].squeeze(axis=dimension)
    ax = plt.gca()
    # Account for non-square pixel sizes. Spacing usually comes from Nifti headers.
    if dimension == 0:
        aspect = spacing[1] / spacing[2]
    elif dimension == 1:
        aspect = spacing[0] / spacing[2]
    else:
        aspect = spacing[0] / spacing[1]
    # This ensures that the coronal and sagittal plot are showing with the head up. For the axial plot (dimension == 0)
    # the default setting of imshow with origin 'upper' is OK.
    origin = 'upper' if dimension == 0 else 'lower'
    ax.imshow(scan_sliced, vmin=np.min(scan), vmax=np.max(scan), cmap='Greys_r', aspect=aspect, origin=origin)
    red = np.ones_like(overlay_sliced)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(red, vmin=0, vmax=1, cmap='Reds', alpha=overlay_sliced, aspect=aspect, origin=origin)

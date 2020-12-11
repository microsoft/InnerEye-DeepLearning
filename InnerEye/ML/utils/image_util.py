#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import label

from InnerEye.Common import common_util
from InnerEye.Common.common_util import any_pairwise_larger
from InnerEye.Common.type_annotations import TupleFloat3, TupleFloat9, TupleInt2, TupleInt3
from InnerEye.ML.config import PaddingMode, SegmentationModelBase
from InnerEye.ML.utils import ml_util

NumpyOrTorch = Union[np.ndarray, torch.Tensor]
Range = Tuple[Union[int, float], Union[int, float]]

# Factor by which array range bounds can be exceeded without triggering an error. If the range is [low, high], we
# only raise an exception if values are outside the range [low-delta, high+delta], where
# delta = (high-low) * VALUE_RANGE_TOLERANCE. Otherwise, we take max with low and min with high, to force all
# values to be inside the bounds.
VALUE_RANGE_TOLERANCE = 1e-6

# The maximum number of classes that a multilabel segmentation map in a HDF5 file can contain.
HDF5_NUM_SEGMENTATION_CLASSES = 10


@dataclass
class ImageHeader:
    """
    A 3D image header
    """
    spacing: TupleFloat3  # Z x Y x X
    origin: TupleFloat3  # X x Y x Z
    direction: TupleFloat9  # X x Y x Z

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


def get_unit_image_header(spacing: Optional[TupleFloat3] = None) -> ImageHeader:
    """
    Creates an ImageHeader object with the origin at 0, and unit direction. The spacing is set to the argument,
    defaulting to (1, 1, 1) if not provided.
    :param spacing: The image spacing, as a (Z, Y, X) tuple.
    """
    if not spacing:
        spacing = (1, 1, 1)
    return ImageHeader(origin=(0, 0, 0), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=spacing)


class ImageDataType(Enum):
    """
    Data type for medical image data (e.g. masks and labels)
    Segmentation maps are one-hot encoded.
    """
    IMAGE = np.float32
    SEGMENTATION = np.float32
    MASK = np.uint8
    CLASSIFICATION_LABEL = np.float32


def apply_mask_to_posteriors(posteriors: NumpyOrTorch, mask: NumpyOrTorch) -> NumpyOrTorch:
    """
    Apply a binary mask to the provided posteriors such that for all voxels outside
    of the mask:
        1) The background class posterior (index == 0) is set to 1.
        2) All other classes posteriors are set to 0.

    :param posteriors: image tensors in shape: Batches (optional) x Classes x Z x Y x X
    :param mask: image tensor in shape: Batches (optional) x Z x Y x X
    :return posteriors with mask applied
    """
    ml_util.check_size_matches(posteriors, mask, matching_dimensions=[-1, -2, -3])

    batch_posteriors = len(posteriors.shape) != 5
    if batch_posteriors:
        posteriors = posteriors[None, ...]
    if len(mask.shape) != 4:
        mask = mask[None, ...]

    if posteriors.shape[0] != mask.shape[0]:
        raise ValueError("posteriors and mask must have the same number of patches, "
                         "found posteriors={}, mask={}".format(posteriors.shape, mask.shape))

    for c in range(posteriors.shape[1]):
        posteriors[:, c, ...][mask == 0] = int(c == 0)

    if batch_posteriors:
        posteriors = posteriors[0]

    return posteriors


def pad_images_for_inference(images: np.ndarray,
                             crop_size: TupleInt3,
                             output_size: Optional[TupleInt3],
                             padding_mode: PaddingMode = PaddingMode.Zero) -> np.ndarray:
    """
    Pad the original image to ensure that the size of the model output as the original image.
    Padding is needed to allow the patches on the corners of the image to be handled correctly, as the model response
    for each patch will only cover the center of  the input voxels for that patch. Hence, add a padding of size
    ceil(output_size - crop_size / 2) around the original image is needed to ensure that the output size of the model
    is the same as the original image size.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param crop_size: the shape of the patches that will be taken from this image.
    :param output_size: the shape of the response for each patch from the model.
    :param padding_mode: a valid numpy padding mode.
    :return: padded copy of the original image.
    """

    def create_padding_vector() -> Tuple[TupleInt2, TupleInt2, TupleInt2]:
        """
        Creates the padding vector.
        """
        diff = np.subtract(crop_size, output_size)
        pad: List[int] = np.ceil(diff / 2.0).astype(int)
        return (pad[0], diff[0] - pad[0]), (pad[1], diff[1] - pad[1]), (pad[2], diff[2] - pad[2])

    if images is None:
        raise Exception("Image must not be none")

    if output_size is None:
        raise Exception("Output size must not be none")

    if not len(images.shape) in [3, 4]:
        raise Exception("Image must be either 3 dimensions (Z x Y x X) or "
                        "Batched into 4 dimensions (Batches x Z x Y x X)")

    if any_pairwise_larger(output_size, crop_size):
        raise Exception("crop_size must be >= output_size, found crop_size:{}, output_size:{}"
                        .format(crop_size, output_size))

    return _pad_images(images=images, padding_vector=create_padding_vector(), padding_mode=padding_mode)


def pad_images(images: np.ndarray,
               output_size: Optional[TupleInt3],
               padding_mode: PaddingMode = PaddingMode.Zero) -> np.ndarray:
    """
    Pad the original images such that their shape after padding is equal to a fixed `output_size`,
    using the provided padding mode.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param output_size: the target output shape after padding.
    :param padding_mode: a valid numpy padding mode
    :return: padded copy of the original image.
    """

    def create_padding_vector() -> Tuple[TupleInt2, TupleInt2, TupleInt2]:
        """
        Creates the padding vector ceil(crop_size - output_size / 2)
        """
        image_spatial_shape = images.shape[-3:]
        diff = np.clip(np.subtract(output_size, image_spatial_shape), a_min=0, a_max=None)
        pad: List[int] = np.ceil(diff / 2.0).astype(int)
        return (pad[0], diff[0] - pad[0]), (pad[1], diff[1] - pad[1]), (pad[2], diff[2] - pad[2])

    if images is None:
        raise Exception("Image must not be none")

    if output_size is None:
        raise Exception("Output size must not be none")

    if not len(images.shape) in [3, 4]:
        raise Exception("Image must be either 3 dimensions (Z x Y x X) or "
                        "Batched into 4 dimensions (Batches x Z x Y x X)")

    return _pad_images(images=images, padding_vector=create_padding_vector(), padding_mode=padding_mode)


def _pad_images(images: np.ndarray,
                padding_vector: Tuple[TupleInt2, TupleInt2, TupleInt2],
                padding_mode: PaddingMode) -> np.ndarray:
    """
    Pad the original images w.r.t to the padding_vector provided for padding on each side in each dimension.

    :param images: the image(s) to be padded, in shape: Z x Y x X or batched in shape: Batches x Z x Y x X.
    :param padding_vector: padding before and after in each dimension eg: ((2,2), (3,3), (2,0))
    will pad 4 pixels in Z (2 on each side), 6 pixels in Y (3 on each side)
    and 2 in X (2 on the left and 0 on the right).
    :param padding_mode: a valid numpy padding mode.
    :return: padded copy of the original image.
    """
    pad_fn = lambda _images: np.stack(
        [np.pad(array=x, pad_width=padding_vector, mode=padding_mode.value) for x in _images])

    # add a batch dimension if required
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        images = pad_fn(images)
        images = np.squeeze(images, axis=0)
    else:
        images = pad_fn(images)

    return images


def posteriors_to_segmentation(posteriors: NumpyOrTorch) -> NumpyOrTorch:
    """
    Perform argmax on the class dimension.

    :param posteriors: Confidence maps [0,1] for each patch per class in format: Batches x Class x Z x Y x X
    or Class x Z x Y x X for non-batched input
    :returns segmentation: argmaxed posteriors with each voxel belonging to a single class: Batches x Z x Y x X
    or Z x Y x X for non-batched input
    """

    if posteriors is None:
        raise ValueError("Posteriors cannot be None.")

    if len(posteriors.shape) < 4 or len(posteriors.shape) > 5:
        raise ValueError(f"Posteriors must have shape: Batches x Class x Z x Y x X or Class x Z x Y x X for "
                         f"non-batched input found {len(posteriors.shape)} dimension(s)")

    # add a batch dimension if required
    argmax_dim = 1 if len(posteriors.shape) == 5 else 0
    if torch.is_tensor(posteriors):
        try:
            segmentation = posteriors.argmax(dim=argmax_dim)
        except RuntimeError:
            # CUDA out of memory, presumably, so we move it to CPU and try again
            posteriors = posteriors.cpu()
            segmentation = posteriors.argmax(dim=argmax_dim)
    else:
        segmentation = np.argmax(posteriors, axis=argmax_dim)
    return segmentation


def largest_connected_components(img: np.ndarray,
                                 deletion_limit: Optional[float],
                                 class_index: Optional[int] = None) -> np.ndarray:
    """
    Select the largest connected binary components (plural) in an image. If deletion_limit is set in which case a
    component is only deleted (i.e. its voxels are False in the output) if its voxel count as a proportion of all the
    True voxels in the input is less than deletion_limit.
    :param img: np.ndarray
    :param deletion_limit: if set, a component is deleted only if its voxel count as a proportion of all the
    True voxels in the input is less than deletion_limit.
    :param class_index: Optional. Can be used to provide a class index for logging purposes if the image contains
     only pixels from a specific class.
    """
    labeled_array, num_features = label(img)
    # When there are lots of labels, which can happen when the predictions are poor, using np.bincount is much
    # quicker than using np.count_nonzero on each label
    component_sizes = np.bincount(labeled_array.flatten())
    # We don't want to count background
    component_sizes[0] = 0
    largest_component_indices: List[Union[int, np.array]] = []
    if deletion_limit is not None and deletion_limit < 0.5:
        # Find the indices of all components with sizes over the threshold - there can be more than one
        # (or there might be none, if all components are small).
        size_threshold = deletion_limit * np.sum(component_sizes)
        largest_component_indices = [idx for (idx, size) in enumerate(component_sizes) if size >= size_threshold]
    if not largest_component_indices:
        # We can get here either if we didn't run the "if" clause above, or if we did but found no components
        # of the required size. In either case, we want to return the largest component, whatever its size.
        largest_component_indices = [np.argmax(component_sizes)]
    out = np.zeros(img.shape, np.bool)
    for idx in largest_component_indices:
        out[labeled_array == idx] = True
    voxels_left = out.sum()
    voxels_pruned = component_sizes.sum() - voxels_left
    if voxels_pruned > 0:
        percent = int(0.5 + 100.0 * voxels_pruned / (voxels_pruned + voxels_left))
        from_class = "" if class_index is None else f" from class {class_index}"
        logging.debug(f"Removed {voxels_pruned} voxels ({percent}%) in {len(component_sizes) - 2} disconnected "
                      f"component(s){from_class}, returning {voxels_left} voxels")
    return out


def extract_largest_foreground_connected_component(
        multi_label_array: np.ndarray,
        restrictions: Optional[List[Tuple[int, Optional[float]]]] = None) -> np.ndarray:
    """
    Extracts the largest foreground connected component per class from a multi-label array.
    :param multi_label_array: An array of class assignments, i.e. value c at (z, y, x) is a class c.
    :param restrictions: restrict processing to a subset of the classes (if provided). Each element is a
    pair (class_index, threshold) where threshold may be None.
    :return: An array of class assignments
    """
    if restrictions is None:
        # process all foreground classes found in the multi_label_array if no restriction is provided
        restrictions = [(x, None) for x in np.unique(multi_label_array) if x > 0]
    elif np.any(np.array([pair[0] for pair in restrictions]) <= 0):
        raise ValueError(f"restrict_foreground_classes must have foreground class indices only. "
                         f"Found {restrictions}")

    result = multi_label_array.copy()

    # update the requested foreground classes such that only the largest connected component is preserved
    for c, threshold in restrictions:
        class_foreground = (result == c)
        if np.any(class_foreground):
            surviving_component = largest_connected_components(class_foreground, threshold, c)
        else:
            surviving_component = class_foreground
        result[surviving_component != class_foreground] = 0

    return result


def merge_masks(masks: np.ndarray) -> np.ndarray:
    """
    Merges a one-hot encoded mask tensor (Classes x Z x Y x X) into a multi-label map with labels corresponding to their
    index in the original tensor of shape (Z x Y x X).
    :param masks: array of shape (Classes x Z x Y x X) containing the mask for each class
    :return: merged_mask of shape (Z x Y x X).
    """
    if masks is None:
        raise Exception("masks must not be None")

    if masks.ndim != 4:
        raise Exception("Expected masks to have 4 dimensions (Classes x Z x Y x X), found: {}".format(masks.ndim))

    merged_mask = np.zeros(shape=masks.shape[-3:])
    for mask_id in range(len(masks)):
        mask = masks[mask_id, ...]
        merged_mask[np.where(mask == 1)] = mask_id

    return merged_mask


def is_binary_array(array: np.ndarray) -> bool:
    """
    Checks to see if the array passed has only binary values.

    :param array: the np.ndarray to check
    :return: True if the Array is binary or False otherwise
    """
    return np.array_equal(array, array.astype(bool))


def multi_label_array_to_binary(array: np.ndarray, num_classes_including_background: int) -> np.ndarray:
    """
    Converts a multimap array into a array of binary masks for each class. If the number of classes is 2,
    the result will contain a binary mask for all entries in the original array where the value was 0,
    and a binary mask for the entries that were 1.

    :param array: An array of class assignments.
    :param num_classes_including_background: The number of class assignments to search for. If 3 classes,
    the class assignments to search for will be 0, 1, and 2.
    :return: an array of size (num_classes_including_background, array.shape)
    """
    return np.stack(list(binaries_from_multi_label_array(array, num_classes_including_background)))


def binaries_from_multi_label_array(array: np.ndarray, num_classes_including_background: int) -> Iterator[np.ndarray]:
    """
    Given multimap array containing C classes, yields an iterator with C elements where each item is a binary array of
    the same shape as the original multimap array. For each binary array, the value is 1 in positions where the value
    of the multimap array is c, and 0 elsewhere.
    """
    for label_index in range(num_classes_including_background):
        yield np.where(array == label_index, 1, 0)


def get_center_crop(image: NumpyOrTorch, crop_shape: TupleInt3) -> NumpyOrTorch:
    """
    Extracts the center region specified by the crop_shape argument from the input image

    :param image: The original image to extract crop from
    :param crop_shape: The shape of the center crop to extract
    :return the center region as specified by the crop_shape argument.
    """
    if image is None or crop_shape is None:
        raise Exception("image and crop_shape must not be None")

    if len(image.shape) != 3 and len(crop_shape) != 3:
        raise Exception("image and crop_shape must have 3 dimensions, found dimensions: image={}, crop_shape={}"
                        .format(len(image.shape), len(crop_shape)))

    if any(np.asarray(crop_shape) > np.asarray(image.shape)):
        raise Exception("crop_shape must be <= to image shape in all dimensions, found shapes: image={}, crop_shape={}"
                        .format(image.shape, crop_shape))

    x, y, z = image.shape
    startx = x // 2 - (crop_shape[0] // 2)
    starty = y // 2 - (crop_shape[1] // 2)
    startz = z // 2 - (crop_shape[2] // 2)
    return image[startx:startx + crop_shape[0], starty:starty + crop_shape[1], startz:startz + crop_shape[2]]


def check_array_range(data: np.ndarray, expected_range: Optional[Range] = None,
                      error_prefix: str = None) -> None:
    """
    Checks if all values in the given array fall into the expected range. If not, raises a
    ValueError, and prints out statistics about the values that fell outside the expected range.
    If no range is provided, it checks that all values in the array are finite (that is, they are not
    infinity and not np.nan

    :param data: The array to check. It can have any size.
    :param expected_range: The interval that all array elements must fall into. The first entry is the lower
    bound, the second entry is the upper bound.
    :param error_prefix: A string to use as the prefix for the error message.
    """
    if expected_range is None:
        valid_pixels = np.isfinite(data)
    else:
        if expected_range[0] > expected_range[1]:
            raise ValueError("The expected range is invalid. The lower bound must be smaller or equal than the"
                             "upper bound, but got: {}".format(expected_range))
        valid_pixels = np.logical_and(data >= expected_range[0], data <= expected_range[1])
        if not np.all(valid_pixels):
            # If values fall outside the range, allow a slightly wider range. If they all fall within that
            # range, force them into the original range. This handles rounding errors which can caus
            # posterior probabilities to be slightly over 1.0.
            range_width = expected_range[1] - expected_range[0]
            delta = range_width * VALUE_RANGE_TOLERANCE
            valid_pixels = np.logical_and(data >= expected_range[0] - delta, data <= expected_range[1] + delta)
            if np.all(valid_pixels):
                data[data < expected_range[0]] = expected_range[0]
                data[data > expected_range[1]] = expected_range[1]
    if not np.all(valid_pixels):
        # All pixel values that are outside the expected interval
        invalid_pixels = data[np.logical_not(valid_pixels)].flatten()
        # Count NaN and Infinity separately
        count_nan = np.count_nonzero(np.isnan(invalid_pixels))
        count_inf = np.count_nonzero(np.isinf(invalid_pixels))
        # Compute the unique pixel values apart from NaN and Inf
        values, counts = np.unique(invalid_pixels[np.isfinite(invalid_pixels)], return_counts=True)
        value_to_count = {values[i]: counts[i] for i in range(values.size)}
        # Maintain a list of invalid values separately, to ensure that we can print NaN and Infinity at the beginning.
        all_invalid_values = []

        def add_to_all_values(value: float, count: int) -> None:
            if count > 0:
                value_to_count[value] = count
                all_invalid_values.append(value)

        add_to_all_values(np.nan, count_nan)
        add_to_all_values(np.inf, count_inf)
        all_invalid_values.extend(values)
        if error_prefix is None:
            error_prefix = ""
        else:
            error_prefix += ": "
        logging.error("{}Invalid values:".format(error_prefix))
        for v, c in value_to_count.items():
            logging.error("{} pixels with value {}".format(c, v))
        print_max = 10
        status = ", ".join(str(v) for v in all_invalid_values[:print_max])
        if invalid_pixels.size > print_max:
            status += ", ... ({} total)".format(invalid_pixels.size)
        range_string = "finite" if expected_range is None else "within [{}, {}]".format(expected_range[0],
                                                                                        expected_range[1])
        raise ValueError("{}All values must be {}. The array contained the following invalid values: {}"
                         .format(error_prefix, range_string, status))


def get_largest_z_slice(mask: np.ndarray) -> int:
    """
    Gets the Z position of the given 3D image that has the largest number of non-zero entries across the whole
    XY plane. If there are multiple Z positions that attain the maximum, return the lowest one.

    :param mask: A 3D image in Z x Y x X order.
    :return: The Z index that has the largest number of non-zero elements across mask[z,:,:]
    """
    if mask.ndim != 3:
        raise ValueError(
            "This code only works with arrays with 3 dimensions, but got an array with shape {}".format(mask.shape))
    # First collapse all dimensions but the leading z dimension into 1. At that point, we can count
    # non-zeros only across the one dimension that is not Z.
    reshaped = mask.reshape((mask.shape[0], mask.size // mask.shape[0]))
    non_zeros = np.count_nonzero(reshaped, axis=1)
    return np.argmax(non_zeros).item()


def one_hot_to_class_indices(labels: torch.Tensor) -> torch.Tensor:
    """
    Converts one hot encoded label tensor to a tensor representing class ids.

    :param labels: One-hot encoded label tensor
    """
    # Check that labels do not overlap with each other
    if not labels.is_floating_point():
        raise TypeError("Input `label` tensor is not a float tensor")

    if not (labels.sum(dim=1) == 1.0).all():
        raise ValueError("Input `label` tensor contains multiple foreground labels for some pixels")

    # Derive class indices
    _, class_ids = labels.max(dim=1)

    return class_ids


def compute_uncertainty_map_from_posteriors(posteriors: np.ndarray) -> np.ndarray:
    """
    Compute voxel wise uncertainty from a given posterior input using
    Normalized Shannon Entropy:  https://en.wiktionary.org/wiki/Shannon_entropy

    :param posteriors: Normalized probability distribution in range [0, 1] for each class,
    in shape: Class x Z x Y x X
    :return: Shannon Entropy for each voxel, shape: Z x Y x X expected range is [0,1] where 1 represents
    low confidence or uniform posterior distribution across classes.
    """
    check_if_posterior_array(posteriors)

    return -np.nansum(posteriors * np.log2(posteriors), axis=0) / np.log2(posteriors.shape[0])


def gaussian_smooth_posteriors(posteriors: np.ndarray, kernel_size_mm: TupleFloat3,
                               voxel_spacing_mm: TupleFloat3) -> np.ndarray:
    """
    Performs Gaussian smoothing on posteriors

    :param posteriors: Normalized probability distribution in range [0, 1] for each class,
    in shape: Class x Z x Y x X
    :param kernel_size_mm: The size of the smoothing kernel in mm to be used in each dimension (Z, Y, X)
    :param voxel_spacing_mm: Voxel spacing to use to map from mm space to pixel space for the
    Gaussian sigma parameter for each dimension in (Z x Y x X) order.
    :return:
    """
    check_if_posterior_array(posteriors)

    if kernel_size_mm is None:
        raise ValueError("kernel_size_mm cannot be None")
    if len(kernel_size_mm) != 3:
        raise ValueError(f"kernel_size_mm must be defined for each dimension Z,Y,X found: {kernel_size_mm}")
    if any([x < 0 for x in kernel_size_mm]):
        raise ValueError(f"kernel_size_mm must be >=0 for each dimension, found: {kernel_size_mm}")

    sigma = np.array(kernel_size_mm) / voxel_spacing_mm
    return np.stack([gaussian_filter(x, sigma=sigma) for x in posteriors], axis=0)


def check_if_posterior_array(posteriors: np.ndarray) -> None:
    """
    Checks if the provided input is a valid posteriors array, raises ValueError otherwise
    """
    if posteriors is None:
        raise Exception("Posteriors cannot be None.")
    if posteriors.ndim != 4:
        raise Exception(f"Posteriors must have shape: Class x Z x Y x X found {len(posteriors.shape)} dimension(s)")
    check_array_range(posteriors, (0.0, 1.0), "Posteriors:")  # type: ignore
    if not np.all(np.isclose(np.sum(posteriors, axis=0), 1)):
        raise ValueError("Posteriors must sum to 1 in the class dimension")


def segmentation_to_one_hot(segmentation: torch.Tensor,
                            use_gpu: bool,
                            result_dtype: torch.dtype) -> torch.Tensor:
    """
    Converts a tensor that contains a segmentation multi-label map to one-hot encoding, running the time-consuming
    operations on the GPU if the use_gpu flag is True. The code assumes that there are no more than
    HDF5_NUM_SEGMENTATION_CLASSES distinct classes in the segmentation.
    For an input tensor of shape [B, C, Z, Y, X] with B batches, C image channels, the result will have size
    [B, C*HDF5_NUM_SEGMENTATION_CLASSES, Z, Y, X]

    :param segmentation: A segmentation as a multi-label map of shape [B, C, Z, Y, X]
    :param use_gpu: If true, and the input is not yet on the GPU, move the intermediate tensors to the GPU. The result
    will be on the same device as the argument `segmentation`
    :param result_dtype: The torch data type that the result tensor should have. This would be either float16 or float32
    :return: A torch tensor with one-hot encoding of the segmentation of shape
    [B, C*HDF5_NUM_SEGMENTATION_CLASSES, Z, Y, X]
    """

    def to_cuda(x: torch.Tensor) -> torch.Tensor:
        if use_gpu and not x.is_cuda:
            return x.cuda()
        else:
            return x

    max_class = torch.max(segmentation).item() + 1
    if max_class > HDF5_NUM_SEGMENTATION_CLASSES:
        raise ValueError(f"The segmentations have a maximum class index of {max_class}, but it must not be "
                         f"more than {HDF5_NUM_SEGMENTATION_CLASSES}")
    input_device = segmentation.device
    # one_hot conversion only works on int64 tensors.
    segmentation = segmentation.to(dtype=torch.int64)

    # Loop over all batches and channels and convert the multimap to one-hot. For 2 input channels and a 10 class
    # map, this would turn a [B, 2, Z, Y, X] multimap tensor into [B, 20, Z, Y, X]
    input_size = segmentation.shape
    result_size = (input_size[0], input_size[1] * HDF5_NUM_SEGMENTATION_CLASSES) + input_size[2:]

    # Pre-allocate the result tensor, to avoid having to store a large list of tensors for stacking.
    result = torch.empty(size=result_size, dtype=result_dtype, device=input_device)
    # Save memory by looping over both batches and channels. This is slightly slower, but saves memory.
    for b in range(segmentation.shape[0]):
        for c in range(segmentation.shape[1]):
            # Ensure that the tensor is on the GPU, then run one-hot
            one_hot_c = torch.nn.functional.one_hot(to_cuda(segmentation[b, c, ...]),
                                                    num_classes=HDF5_NUM_SEGMENTATION_CLASSES)
            # Convert from 64 bit integer to 16 bit float to save memory
            one_hot_c = one_hot_c.to(dtype=result_dtype)
            # one_hot adds the class dimension at the end. Convert such that each segmentation is equivalent to
            # an image channel.
            one_hot_c = one_hot_c.permute(3, 0, 1, 2)
            c_start = c * HDF5_NUM_SEGMENTATION_CLASSES
            c_end = c_start + HDF5_NUM_SEGMENTATION_CLASSES
            result[b, c_start:c_end] = one_hot_c
    return result


def get_class_weights(target: torch.Tensor, class_weight_power: float = 1.0) -> torch.Tensor:
    """
    Returns class weights inversely proportional to some power of the number of pixels in each class.

    :param target: one-hot tensor of shape (B, C, Z, X, Y); thus class dimension (of size C) is dimension 1
    :param class_weight_power: power to raise 1/c to, for each class count c
    """
    with torch.no_grad():
        class_counts = target.sum([0] + list(range(2, target.dim()))).float()  # sum over all except class dimension
        class_counts[class_counts == 0.0] = 1.0  # prevent 1/0 when invert - value doesn't matter if no voxels
        class_weights = class_counts ** (-class_weight_power)
        # Normalize so mean of class weights is 1.0
        class_weights *= class_weights.shape[0] / class_weights.sum()
    return class_weights


def apply_slice_exclusion_rules(model_config: SegmentationModelBase,
                                segmentation: np.ndarray) -> np.ndarray:
    """
    Applies each slice exclusion rule to segmentation, modifying it in place.
    """

    if model_config.slice_exclusion_rules is None:
        return segmentation

    for rule in model_config.slice_exclusion_rules:
        rule.validate(model_config.ground_truth_ids)

        # assume label indices start from 1, 0 is the background
        higher_class_label = model_config.ground_truth_ids.index(rule.higher_class) + 1
        lower_class_label = model_config.ground_truth_ids.index(rule.lower_class) + 1

        slices_with_higher_class = np.nonzero(np.any(segmentation == higher_class_label, axis=(1, 2)))[0]
        slices_with_lower_class = np.nonzero(np.any(segmentation == lower_class_label, axis=(1, 2)))[0]

        if slices_with_higher_class.shape[0] != 0 and slices_with_lower_class.shape[0] != 0:
            # the origin for the segmentation images are at the bottom of the image
            # find lowest z index in which a segmentation label of higher_class exists:
            lowest_index_higher_class = slices_with_higher_class[0]
            # find highest z index in which a segmentation label of lower_class exists
            highest_index_lower_class = slices_with_lower_class[-1]

            overlap = lowest_index_higher_class <= highest_index_lower_class

            if overlap:

                overlapping_region = segmentation[lowest_index_higher_class:highest_index_lower_class + 1]

                dominant_class = rule.higher_class if rule.higher_dominates else rule.lower_class
                replaced_class = rule.lower_class if rule.higher_dominates else rule.higher_class
                dominant_class_label = higher_class_label if rule.higher_dominates else lower_class_label
                replaced_class_label = lower_class_label if rule.higher_dominates else higher_class_label

                replace_locations = (overlapping_region == replaced_class_label)
                voxels_replaced = replace_locations.sum()
                overlapping_region[replace_locations] = dominant_class_label

                if lowest_index_higher_class == highest_index_lower_class:
                    logging.debug(f"Slice exclusion: in slice {lowest_index_higher_class}, "
                                  f"replaced {voxels_replaced} {replaced_class} voxels "
                                  f"with {dominant_class}")
                else:
                    logging.debug(f"Slice exclusion: in slices "
                                  f"{lowest_index_higher_class}-{highest_index_lower_class}, "
                                  f"replaced {voxels_replaced} {replaced_class} voxels "
                                  f"with {dominant_class}")

    return segmentation


def find_intersection_array_indices(indices1: Union[np.ndarray, Tuple[np.ndarray, ...]],
                                    indices2: Union[np.ndarray, Tuple[np.ndarray, ...]],
                                    shape: Union[np.ndarray, Tuple[int, ...]]) -> Tuple[np.ndarray, ...]:
    """
    Finds the intersection of two sets of indices for multidimensional arrays

    :param indices1: Tuple with n arrays, each containing indices in the ith dimension, as returned by np.where()
    :param indices2: Tuple with n arrays, each containing indices in the ith dimension, as returned by np.where()
    :param shape: The shape of the array that was indexed originally
    :return: Tuple with n arrays, each containing indices in the ith dimension
    """

    if len(indices1) != len(indices2) or len(indices1) != len(shape):
        raise ValueError("find_intersection_array_indices: "
                         "Trying to compare indices from incompatible array shapes")
    row_major_indices1 = np.ravel_multi_index(indices1, shape)
    row_major_indices2 = np.ravel_multi_index(indices2, shape)
    intersection_in_row_major = np.intersect1d(row_major_indices1, row_major_indices2, assume_unique=True)
    intersection_indices = np.unravel_index(intersection_in_row_major, shape)
    return intersection_indices


def apply_summed_probability_rules(model_config: SegmentationModelBase,
                                   posteriors: NumpyOrTorch,
                                   segmentation: NumpyOrTorch) -> NumpyOrTorch:
    """
    Applies summed probability rules to segmentation, modifying it in place.

    :param model_config: Model configuration information
    :param posteriors: Confidences per voxel per class, in format Batch x Classes x Z x Y x X if batched,
                       or Classes x Z x Y x X if not batched.
    :param segmentation: Class labels per voxel, in format Batch x Z x Y x X if batched, or Z x Y x X if not batched.
    :return: Modified segmentation, as Batch x Z x Y x X if batched, or Z x Y x X if not batched.
    """

    if not model_config.summed_probability_rules:
        return segmentation

    if posteriors is None:
        raise ValueError("summed_probability_rules: Posteriors cannot be None.")
    if segmentation is None:
        raise ValueError("summed_probability_rules: Segmentation cannot be None.")
    if posteriors.ndim < 4 or posteriors.ndim > 5:  # type: ignore
        raise ValueError(f"summed_probability_rules: Posteriors must have shape: "
                         f"Batches x Class x Z x Y x X or Class x Z x Y x X for non-batched input "
                         f"found {len(posteriors.shape)} dimension(s)")
    if posteriors.ndim - 1 != segmentation.ndim or posteriors.shape[-3:] != segmentation.shape[-3:]:  # type: ignore
        raise ValueError(f"summed_probability_rules: Posteriors and segmentation have incompatible shapes: "
                         f"{posteriors.shape} and {segmentation.shape}")
    if posteriors.ndim == 5 and posteriors.shape[0] != segmentation.shape[0]:  # type: ignore
        raise ValueError("summed_probability_rules: Posteriors and segmentation have different batch sizes")

    if model_config.summed_probability_rules is not None:
        for rule in model_config.summed_probability_rules:

            rule.validate(model_config.ground_truth_ids)

            # assume label indices start from 1, 0 is the background
            first_class_label = model_config.ground_truth_ids.index(rule.first_class) + 1
            second_class_label = model_config.ground_truth_ids.index(rule.second_class) + 1
            external_class_label = model_config.ground_truth_ids.index(rule.external_class) + 1

            if posteriors.ndim == 5:  # type: ignore
                replace_indices = np.where(posteriors[:, external_class_label]
                                           < posteriors[:, [first_class_label, second_class_label]].sum(1))
                first_class_indices = np.where(posteriors[:, first_class_label] >= posteriors[:, second_class_label])
                second_class_indices = np.where(posteriors[:, first_class_label] < posteriors[:, second_class_label])
            else:
                replace_indices = np.where(posteriors[external_class_label]
                                           < posteriors[[first_class_label, second_class_label]].sum(0))
                first_class_indices = np.where(posteriors[first_class_label] >= posteriors[second_class_label])
                second_class_indices = np.where(posteriors[first_class_label] < posteriors[second_class_label])

            external_indices = np.where(segmentation == external_class_label)

            input_shape = segmentation.shape
            replace_indices = find_intersection_array_indices(replace_indices, external_indices, input_shape)
            replace_indices_first_class = find_intersection_array_indices(replace_indices, first_class_indices,
                                                                          input_shape)
            replace_indices_second_class = find_intersection_array_indices(replace_indices, second_class_indices,
                                                                           input_shape)

            logging.debug(f"Summed probability rules: Replacing {replace_indices_first_class[0].shape[0]} "
                          f"{rule.external_class} voxels with class {rule.first_class}")
            logging.debug(f"summed_probability_rules: Replacing {replace_indices_second_class[0].shape[0]} "
                          f"{rule.external_class} voxels with class {rule.second_class}")

            segmentation[replace_indices_first_class] = first_class_label
            segmentation[replace_indices_second_class] = second_class_label

    return segmentation

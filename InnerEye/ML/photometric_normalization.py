#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import Any, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.filters import threshold_otsu

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase
from InnerEye.ML.dataset.sample import Sample
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.utils.image_util import check_array_range
from InnerEye.ML.utils.transforms import CTRange, LinearTransform, Transform3D


class WindowNormalizationForScalarItem(Transform3D[ScalarItem]):
    """
    Transform3D to apply window normalization to "images" of a ScalarItem.
    """
    # noinspection PyMissingConstructor
    def __init__(self,
                 output_range: Tuple[float, float] = (0, 1),
                 sharpen: float = 1.9,
                 tail: float = 1.0) -> None:
        """
        :param output_range: The desired value range of the result image.
        :param sharpen: number of standard deviation either side of mean to include in the window
        :param tail: Default 1, allow window range to include more of tail of distribution.
        """
        self.output_range = output_range
        self.sharpen = sharpen
        self.tail = tail

    def __call__(self, item: ScalarItem) -> ScalarItem:
        return item.clone_with_overrides(
            images=torch.tensor(mri_window(image_in=item.images.numpy(),
                                           output_range=self.output_range,
                                           mask=None,
                                           sharpen=self.sharpen,
                                           tail=self.tail)[0],
                                dtype=item.images.dtype,
                                device=item.images.device)
        )


class PhotometricNormalization(Transform3D[Sample]):
    def __init__(self, config_args: SegmentationModelBase = None, **params: Any):
        super().__init__(**params)
        if config_args is None:
            self.norm_method = PhotometricNormalizationMethod.Unchanged
            return

        if config_args is not None:
            self.norm_method = config_args.norm_method
            self.output_range = config_args.output_range
            self.level = config_args.level
            self.window = config_args.window
            self.debug_mode = config_args.debug_mode
            self.tail = config_args.tail
            self.sharpen = config_args.sharpen
            self.trim_percentiles = config_args.trim_percentiles
            self.status_of_most_recent_call: Optional[str] = None

    def __call__(self, sample: Sample) -> Sample:
        return sample.clone_with_overrides(
            image=self.transform(
                image=sample.image,
                mask=sample.mask,
                patient_id=sample.patient_id
            )
        )

    def transform(self, image: Union[np.ndarray, torch.Tensor],
                  mask: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  patient_id: Optional[int] = None) -> Union[np.ndarray, torch.Tensor]:
        if mask is None:
            if torch.is_tensor(image):
                mask = torch.ones_like(image)
            else:
                mask = np.ones_like(image)

        self.status_of_most_recent_call = None
        if self.norm_method == PhotometricNormalizationMethod.Unchanged:
            image_out = image
        elif self.norm_method == PhotometricNormalizationMethod.SimpleNorm:
            image_out = simple_norm(image, mask, self.debug_mode)
        elif self.norm_method == PhotometricNormalizationMethod.MriWindow:
            if self.sharpen is None:
                raise ValueError("The 'sharpen' parameter must be provided.")
            if not (isinstance(self.tail, list) or isinstance(self.tail, float)):
                raise ValueError(
                    "The 'tail' parameter must be provided and set to a float value or a list of float values.")
            image_out, status = mri_window(
                image, mask,
                self.output_range, self.sharpen, self.tail, self.debug_mode
            )
            self.status_of_most_recent_call = status
        elif self.norm_method == PhotometricNormalizationMethod.CtWindow:
            if self.level is None:
                raise ValueError("The 'level' parameter must be provided.")
            if self.window is None:
                raise ValueError("The 'window' parameter must be provided.")
            image_out = CTRange.transform(data=image, output_range=self.output_range,
                                          level=self.level, window=self.window, use_gpu=self.use_gpu)
        elif self.norm_method == PhotometricNormalizationMethod.TrimmedNorm:
            image_out, status = normalize_trim(image, mask,  # type: ignore
                                               self.output_range, self.sharpen, self.trim_percentiles,
                                               self.debug_mode)
            self.status_of_most_recent_call = status
        else:
            raise ValueError("Unknown normalization method {}".format(self.norm_method))
        if patient_id is not None and self.status_of_most_recent_call is not None:
            logging.debug(f"Photonorm patient {patient_id}: {self.status_of_most_recent_call}")
        check_array_range(image_out, error_prefix="Normalized image")

        return image_out


def simple_norm(image_in: np.ndarray, mask: np.ndarray, debug_mode: bool = False) -> np.ndarray:
    """
    Normalizes a single image to have mean 0 and standard deviation 1

    :param image_in: image to normalize
    :param mask: image, has W x H x D
    :param debug_mode: whether to log means and SDs
    :return: normalized image
    """
    if not np.issubdtype(image_in.dtype, np.floating):
        raise Exception("normalize::simple_norm: Input image is not a floating type")

    image_shape = np.shape(image_in)
    nchannel = image_shape[0]
    iout = np.zeros(image_shape, dtype=image_in.dtype)

    for ichannel in range(nchannel):

        i = image_in[ichannel, ...].flatten()
        m = mask.flatten()
        if debug_mode:
            logging.info(" In norm before:  Standard Deviation, Mean ,{0: 4.1f}, {1: 4.1f}".format(np.std(i[m == 1]),
                                                                                                   np.mean(i[m == 1])))
        mean_i = np.mean(i[m == 1])
        std_i = np.std(i[m == 1])
        i = i - mean_i
        i = i / std_i
        iout[ichannel, ...] = i.reshape(image_shape[1:])
        if debug_mode:
            logging.info(" In norm after:  Standard Deviation, Mean ,{0: 4.1f}, {1: 4.1f}".format(np.std(i[m == 1]),
                                                                                                  np.mean(i[m == 1])))

    return iout


def normalize_trim(image: np.ndarray,
                   mask: np.ndarray,
                   output_range: Tuple[float, float] = (-1.0, 1.0),
                   sharpen: float = 1.9,
                   trim_percentiles: Tuple[float, float] = (2.0, 98.0),
                   debug_mode: bool = False) -> np.ndarray:
    """
    Normalizes a single image to have mean 0 and standard deviation 1
    Normalising occurs after percentile thresholds have been applied to strip out extreme values

    :param image: The image to normalize, size Channels x Z x Y x X
    :param mask: Consider only pixel values of the input image for which the mask is non-zero. Size Z x Y x X
    :param output_range: The desired value range of the result image.
    :param sharpen: number of standard deviation either side of mean to include in the window.
    :param trim_percentiles: Only consider voxel values between those two percentiles when computing mean and std.
    :param debug_mode: If true, create a diagnostic plot (interactive)
    :return: trimmed-normalized image
    """

    image_shape = image.shape
    imout = np.zeros_like(image)
    in_mask = mask > 0.5
    status = ""
    for ichannel in range(image_shape[0]):
        if ichannel > 0:
            status += "Channel {}: ".format(ichannel)
        channel_image = image[ichannel, ...]
        pixels_inside_mask = channel_image[in_mask].flatten().astype(float)
        # First remove all values that fall outside the trim_percentiles
        thresholds = np.percentile(pixels_inside_mask, trim_percentiles, interpolation='midpoint')
        lower_threshold = thresholds[0]
        upper_threshold = thresholds[1]
        above_lower = pixels_inside_mask > lower_threshold
        below_upper = pixels_inside_mask < upper_threshold
        inside_thresholds = np.logical_and(above_lower, below_upper)
        # Compute robust statistics off the pixel values that are inside the trim values
        median, estimated_std, min_value, max_value = robust_mean_std(pixels_inside_mask[inside_thresholds])
        # Compute an input value range from median and robust std, going as many standard deviations
        # as specified by the sharpen parameter
        input_range = (max(median - estimated_std * sharpen, min_value),
                       min(median + estimated_std * sharpen, max_value))
        # Use Polynomial transform function to convert data to output range. This also sets values outside
        # the input_range to the boundary values.
        channel_output = LinearTransform.transform(
            data=channel_image,
            input_range=input_range,
            output_range=output_range
        )
        channel_output[np.logical_not(in_mask)] = output_range[0]
        imout[ichannel, ...] = channel_output
        status += "Range ({0:0.0f}, {1:0.0f}) ".format(input_range[0], input_range[1])
        logging.info(status)
        if debug_mode:
            print('median, estimated_std', median, estimated_std)
            #
            # Normalise values to zero mean and unit variance
            #
            fig, axs = plt.subplots(2, 2, figsize=(9, 9))
            axs[0, 0].set_title("Original Image")
            axs[0, 0].imshow(image[0, :, :, 70], cmap='gray')
            # axs[1,0].hist(image.flatten(), 100)
            axs[1, 0].set_title("Original Image - Histogram with Mask")
            axs[1, 0].set_xlim(lower_threshold, upper_threshold)
            axs[1, 0].hist(channel_image[in_mask].flatten(), 20)
            axs[0, 1].set_title("Normalised Image, Level= {level:4.1f},\n "
                                "Window range {in1} to {in2}".format(level=median, in1=lower_threshold,
                                                                     in2=upper_threshold))
            axs[0, 1].imshow(imout[0, :, :, 70], cmap='gray')
            axs[1, 1].set_title("Normalised Image - Histogram with Mask")
            axs[1, 1].hist(channel_image[in_mask], 20)
            plt.show()
    return imout, status


def robust_mean_std(data: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Computes robust estimates of mean and standard deviation in the given array.
    The median is the robust estimate for the mean, the standard deviation is computed from the
    inter-quartile ranges.
    :param data: The data for which mean and std should be computed.
    :return: A 4-tuple with values (median, robust_std, minimum data value, maximum data value)
    """
    if data.ndim != 1:
        data = data.flatten()
    quartiles = np.percentile(data, (0, 25, 50, 75, 100), interpolation='midpoint')
    min_value = quartiles[0]
    quart25 = quartiles[1]
    median = quartiles[2]
    quart75 = quartiles[3]
    max_value = quartiles[4]
    # Estimate standard deviation from inter quartile range:
    # Quartile 1 is at -0.67 of the standard normal (Excel NORMSINV(0.25))
    # Quartile 3 is at 0.67 of the standard normal (Excel NORMSINV(0.75))
    # Inter quartile range hence spans 2 * 0.67 standard deviations
    std = (quart75 - quart25) / (2 * 0.67448975)
    return median, std, min_value, max_value


def mri_window(image_in: np.ndarray,
               mask: Optional[np.ndarray],
               output_range: Tuple[float, float] = (-1.0, 1.0),
               sharpen: float = 1.9,
               tail: Union[List[float], float] = 1.0,
               debug_mode: bool = False) -> Tuple[np.ndarray, str]:
    """
    This function takes an MRI Image,  removes to first peak of values (air). Then a window range is found centered
    around the mean of the remaining values and with a range controlled by the standard deviation and the sharpen
    input parameter. The larger sharpen is, the wider the range. The resulting values are the normalised to the given
    output_range, with values below and above the range being set the the boundary values.
    :param image_in: The image to normalize.
    :param mask: Consider only pixel values of the input image for which the mask is non-zero. If None the whole
    image is considered.
    :param output_range: The desired value range of the result image.
    :param sharpen: number of standard deviation either side of mean to include in the window
    :param tail: Default 1, allow window range to include more of tail of distribution.
    :param debug_mode: If true, create diagnostic plots.
    :return: normalized image
    """
    nchannel = image_in.shape[0]
    imout = np.zeros_like(image_in)
    if isinstance(tail, int):
        tail = float(tail)
    if isinstance(tail, float):
        tail = [tail] * nchannel
    status = ""
    for ichannel in range(nchannel):
        if ichannel > 0:
            status += "Channel {}: ".format(ichannel)
        # Flatten to apply Otsu_thresholding
        imflat = image_in[ichannel, ...].flatten()
        if mask is None:
            maflat = None
            in_mask = False
        else:
            maflat = mask.flatten()
            in_mask = mask > 0  # type: ignore
        # Find Otsu's threshold for the values of the input image
        threshold = threshold_otsu(imflat)
        # Find window level
        level, std_i, _, max_foreground = robust_mean_std(imflat[imflat > threshold])
        # If lower value of window is below threshold replace lower value with threshold
        input_range = (max(level - std_i * sharpen, threshold),
                       min(max_foreground, level + tail[ichannel] * std_i * sharpen))
        im_thresh = image_in[ichannel, ...]
        im_thresh[image_in[ichannel, ...] < threshold] = 0
        # Use Polynomial transform function to convert data to output range
        imout[ichannel, ...] = LinearTransform.transform(im_thresh, input_range, output_range)
        status += f"Otsu {threshold:0.0f}, level {level:0.0f}, range ({input_range[0]:0.0f}, {input_range[1]:0.0f}) "
        logging.debug(status)
        if debug_mode:
            print('Otsu {}, range {}'.format(threshold, input_range))
            if mask is None:
                no_thresh = np.sum(imflat < threshold)
                no_high = np.sum(imout == output_range[1])
                pc_thresh = no_thresh / np.numel(imflat) * 100  # type: ignore
                pc_high = no_high / np.numel(imflat) * 100  # type: ignore
            else:
                no_thresh = np.sum(imflat[maflat == 1] < threshold)
                no_high = np.sum(imout == output_range[1])
                pc_thresh = no_thresh / np.sum(in_mask) * 100
                pc_high = no_high / np.sum(in_mask) * 100

            print('Percent of values outside window range: low,high', pc_thresh, pc_high, no_high)

            with open("channels_trim.txt", 'a') as fileout:
                fileout.write("Thresholded: {ich:d}, {pl:4.2f}, {ph:4.2f} \n".format(ich=ichannel,
                                                                                     pl=pc_thresh,
                                                                                     ph=pc_high))

            # Plot input distribution
            fig, axs = plt.subplots(2, 2, figsize=(9, 9))
            axs[0, 0].set_title("Original Image")
            axs[0, 0].imshow(image_in[ichannel, :, :, 70], cmap='gray')
            #    axs[1,0].hist(image.flatten(), 100)
            axs[1, 0].set_title("Original Image - Histogram with Mask")
            if mask is None:
                axs[1, 0].hist(image_in[ichannel, ...].flatten(), 200)
            else:
                axs[1, 0].hist(image_in[ichannel, ...][in_mask].flatten(), 200)
            axs[0, 1].set_title("Normalised Image, Level= {level:4.1f},\n "
                                "Window range {in1:4.1f} to {in2:4.1f}, \n"
                                "{pt:4.1f} % below threshold, {ph:4.1f} % above window \n"
                                "Threshold= {th:4.1f}"
                                .format(level=level, in1=input_range[0], in2=input_range[1], pt=pc_thresh,
                                        ph=pc_high, th=threshold))
            axs[0, 1].imshow(imout[ichannel, :, :, 70], cmap='gray')
            axs[1, 1].set_title("Normalised Image - Histogram with Mask")
            if mask is None:
                axs[1, 1].hist(imout[ichannel, ...].flatten(), 200)
            else:
                axs[1, 1].hist(imout[ichannel, ...][in_mask].flatten(), 200)
            plt.show()

    return imout, status

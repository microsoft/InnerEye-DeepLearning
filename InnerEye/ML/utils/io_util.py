#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from copy import copy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Generic, Iterable, List, Optional, Tuple, Type, TypeVar, Union
from skimage.transform import resize

import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from tabulate import tabulate

from InnerEye.Common import common_util
from InnerEye.Common.type_annotations import PathOrString, TupleFloat3, TupleInt3
from InnerEye.ML.config import DEFAULT_POSTERIOR_VALUE_RANGE, PhotometricNormalizationMethod, \
    SegmentationModelBase
from InnerEye.ML.dataset.sample import PatientDatasetSource, Sample
from InnerEye.ML.utils.hdf5_util import HDF5Object
from InnerEye.ML.utils.image_util import ImageDataType, ImageHeader, check_array_range, get_center_crop, is_binary_array
from InnerEye.ML.utils.transforms import LinearTransform, get_range_for_window_level

RESULTS_POSTERIOR_FILE_NAME_PREFIX = "posterior_"
RESULTS_SEGMENTATION_FILE_NAME_PREFIX = "segmentation"
TensorOrNumpyArray = TypeVar('TensorOrNumpyArray', torch.Tensor, np.ndarray)


@dataclass
class ImageWithHeader:
    """
    A 3D image with header
    """
    image: np.ndarray  # Z x Y x X
    header: ImageHeader

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


class MedicalImageFileType(Enum):
    """
    Supported types of medical image formats
    """
    NIFTI_COMPRESSED_GZ = ".nii.gz"
    NIFTI = ".nii"


class NumpyFile(Enum):
    """
    Supported file extensions that indicate Numpy data.
    """
    Numpy = ".npy"


class HDF5FileType(Enum):
    """
    Supported file extensions that indicate HDF5 data.
    """
    HDF5 = ".h5"
    HDF5_EXPLICIT = ".hdf5"
    HDF5_COMPRESSED_GZ = ".h5.gz"
    HDF5_COMPRESSED_SZ = ".h5.sz"


class DicomFileType(Enum):
    """
    Supported file extensions that indicate Dicom data.
    """
    Dicom = ".dcm"


VALID_NIFTI_EXTENSIONS_TUPLE = tuple([f.value for f in MedicalImageFileType])
VALID_HDF5_EXTENSIONS_TUPLE = tuple([f.value for f in HDF5FileType])
VALID_NUMPY_EXTENSIONS_TUPLE = tuple([f.value for f in NumpyFile])
VALID_DICOM_EXTENSIONS_TUPLE = tuple([f.value for f in DicomFileType])


def _file_matches_extension(file: PathOrString, valid_extensions: Iterable[str]) -> bool:
    """
    Returns true if the given file name has any of the provided file extensions.

    :param file: The file name to check.
    :param valid_extensions: A tuple with all the extensions that are considered valid.
    :return: True if the file has any of the given extensions.
    """
    dot = "."
    extensions_with_dot = tuple(e if e.startswith(dot) else dot + e for e in valid_extensions)
    return str(file).lower().endswith(extensions_with_dot)


def is_nifti_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a compressed or uncompressed
    Nifti file. This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Nifti file.
    """
    return _file_matches_extension(file, VALID_NIFTI_EXTENSIONS_TUPLE)


def is_numpy_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a Numpy file.
    This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Numpy file.
    """
    return _file_matches_extension(file, VALID_NUMPY_EXTENSIONS_TUPLE)


def is_hdf5_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a compressed or uncompressed
    HDF5 file. This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a HDF5 file.
    """
    return _file_matches_extension(file, VALID_HDF5_EXTENSIONS_TUPLE)


def is_dicom_file_path(file: PathOrString) -> bool:
    """
    Returns true if the given file name appears to belong to a Dicom file.
    This is done based on extensions only. The file does not need to exist.

    :param file: The file name to check.
    :return: True if the file name indicates a Dicom file.
    """
    return _file_matches_extension(file, VALID_DICOM_EXTENSIONS_TUPLE)


def read_image_as_array_with_header(file_path: Path) -> Tuple[np.ndarray, ImageHeader]:
    """
    Read image with simpleITK as a ndarray.

    :param file_path:
    :return: Tuple of ndarray with image in Z Y X and Spacing in Z X Y
    """
    image: sitk.Image = sitk.ReadImage(str(file_path))
    img = sitk.GetArrayFromImage(image)
    spacing = reverse_tuple_float3(image.GetSpacing())
    # We keep origin and direction on the original shape since it is not used in this library
    # only for saving images correctly
    origin = image.GetOrigin()
    direction = image.GetDirection()

    return img, ImageHeader(origin=origin, direction=direction, spacing=spacing)


def load_nifti_image(path: PathOrString, image_type: Optional[Type] = float) -> ImageWithHeader:
    """
    Loads a single .nii, or .nii.gz image from disk. The image to load must be 3D.

    :param path: The path to the image to load.
    :return: A numpy array of the image and header data if applicable.
    :param image_type: The type to load the image in, set to None to not cast, default is float
    :raises ValueError: If the path is invalid or the image is not 3D.
    """

    def _is_valid_image_path(_path: Path) -> bool:
        """
        Validates a path for an image. Image must be .nii, or .nii.gz.
        :param _path: The path to the file.
        :return: True if it is valid, False otherwise
        """
        if _path.is_file():
            return is_nifti_file_path(_path)
        return False

    if isinstance(path, str):
        path = Path(path)
    if path is None or not _is_valid_image_path(path):
        raise ValueError("Invalid path to image: {}".format(path))

    img, header = read_image_as_array_with_header(path)

    # ensure a 3D image is loaded
    if not len(img.shape) == 3:
        raise ValueError("The loaded image should be 3D (image.shape: {})".format(img.shape))

    if image_type is not None:
        img = img.astype(dtype=image_type)

    return ImageWithHeader(image=img, header=header)


def load_numpy_image(path: PathOrString) -> np.ndarray:
    """
    Loads an array from a numpy file.
    :param path: The path to the numpy file.
    """
    image = np.load(path)
    return image


def load_dicom_image(path: PathOrString) -> np.ndarray:
    """
    Loads an array from a single dicom file.
    :param path: The path to the dicom file.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    image = reader.Execute()
    pixels = sitk.GetArrayFromImage(image)
    # Return a float array, we may resize this in load_3d_images_and_stack, and interpolation will not work on int
    return pixels.astype(np.float)


def load_hdf5_file(path_str: Union[str, Path], load_segmentation: bool = False) -> HDF5Object:
    """
    Loads a single HDF5 file.
    :param path_str: The path of the HDF5 file that should be loaded.
    :param load_segmentation: If True, the `segmentation` field of the result object will be populated. If
    False, the field will be set to None.
    :return: numpy array
    """

    def _is_valid_hdf5_path(_path: Path) -> bool:
        """
        Validates a path for an image
        :param _path:
        :return:
        """
        return _path.is_file() and is_hdf5_file_path(_path)

    path = Path(path_str)

    if path is None or not _is_valid_hdf5_path(path):
        raise ValueError(f"Invalid path: {path}")

    return HDF5Object.from_file(path, load_segmentation=load_segmentation)


@dataclass(frozen=True)
class ImageAndSegmentations(Generic[TensorOrNumpyArray]):
    images: TensorOrNumpyArray
    segmentations: Optional[TensorOrNumpyArray] = None


def load_images_and_stack(files: Iterable[Path],
                             load_segmentation: bool,
                             center_crop_size: Optional[TupleInt3] = None,
                             image_size: Optional[TupleInt3] = None) -> ImageAndSegmentations[torch.Tensor]:
    """
    Attempts to load a set of files, all of which are expected to contain 3D images of the same size (Z, X, Y)
    They are all stacked along dimension 0 and returned as a torch tensor of size (B, Z, X, Y)
    Images are returned as torch.float32 tensors, segmentations are returned as torch.uint8 tensors (multimaps).

    :param files: The paths of the files to load.
    :param load_segmentation: If True it loads segmentation if present on the same file as the image. This is only
    supported for loading from HDF5 files.
    :param center_crop_size: If supplied, all loaded images will be cropped to the size given here. The crop will be
    taken from the center of the image.
    :param image_size: If supplied, all loaded images will be resized immediately after loading.
    :return: A wrapper class that contains the loaded images, and if load_segmentation is True, also the segmentations
    that were present in the files.
    """
    images = []
    segmentations = []

    def from_numpy_crop_and_resize(array: np.ndarray) -> torch.Tensor:
        if image_size:
            if not issubclass(array.dtype.type, np.floating):
                raise ValueError("Array must be of type float.")
            if array.shape[0] == 1 and not image_size[0] == 1:
                raise ValueError(f"Input image is 2D with singleton dimension {array.shape}, but parameter "
                                 f"image_shape has non-singleton first dimension {image_size}")
            array = resize(array, image_size, anti_aliasing=True)
        t = torch.from_numpy(array)
        if center_crop_size:
            if array.shape[0] == 1 and not center_crop_size[0] == 1:
                raise ValueError(f"Input image is 2D with singleton dimension {array.shape}, but parameter "
                                 f"center_crop_size has non-singleton first dimension {center_crop_size}")
            return get_center_crop(t, center_crop_size)
        return t

    for file_path in files:
        image_and_segmentation = load_image_in_known_formats(file_path, load_segmentation)
        image_numpy = image_and_segmentation.images

        if image_numpy.ndim == 4 and image_numpy.shape[0] == 1:
            image_numpy = image_numpy.squeeze(axis=0)
        elif image_numpy.ndim == 2:
            image_numpy = image_numpy[None, ...]
        elif image_numpy.ndim != 3:
            raise ValueError(f"Image {file_path} has unsupported shape: {image_numpy.shape}")

        images.append(from_numpy_crop_and_resize(image_numpy))
        if load_segmentation:
            # Segmentations are loaded as UInt8. Convert to one-hot encoding as late as possible,
            # that is only before feeding into the model
            segmentations.append(from_numpy_crop_and_resize(image_and_segmentation.segmentations))

    image_tensor = torch.stack(images, dim=0) if len(images) > 0 else torch.empty(0)
    segmentation_tensor = torch.stack(segmentations, dim=0) if len(segmentations) > 0 else torch.empty(0)
    return ImageAndSegmentations(images=image_tensor, segmentations=segmentation_tensor)


def load_image_in_known_formats(file: Path,
                                load_segmentation: bool) -> ImageAndSegmentations[np.ndarray]:
    """
    Loads an image from a file in the given path. At the moment, this supports Nifti, HDF5, numpy and dicom files.

    :param file: The path of the file to load.
    :param load_segmentation: If True it loads segmentation if present on the same file as the image.
    :return: a wrapper class that contains the images and segmentation if present
    """
    if is_hdf5_file_path(file):
        hdf5_object = load_hdf5_file(path_str=file,
                                     load_segmentation=load_segmentation)
        return ImageAndSegmentations(images=hdf5_object.volume,
                                     segmentations=hdf5_object.segmentation if load_segmentation else None)
    elif is_nifti_file_path(file):
        return ImageAndSegmentations(images=load_nifti_image(path=file).image)
    elif is_numpy_file_path(file):
        return ImageAndSegmentations(images=load_numpy_image(path=file))
    elif is_dicom_file_path(file):
        return ImageAndSegmentations(images=load_dicom_image(path=file))
    else:
        raise ValueError(f"Unsupported image file type for path {file}")


def load_labels_from_dataset_source(dataset_source: PatientDatasetSource) -> np.ndarray:
    """
    Load labels containing segmentation binary labels in one-hot-encoding.
    In the future, this function will be used to load global class and non-imaging information as well.

    :param dataset_source: The dataset source for which channels are to be loaded into memory.
    :return A label sample object containing ground-truth information.
    """
    labels = np.stack(
        [load_nifti_image(gt, ImageDataType.SEGMENTATION.value).image for gt in dataset_source.ground_truth_channels])

    # Add the background binary map
    background = np.ones_like(labels[0])
    for c in range(len(labels)):
        background[labels[c] == 1] = 0
    background = background[None, ...]
    return np.vstack((background, labels))


def load_images_from_dataset_source(dataset_source: PatientDatasetSource) -> Sample:
    """
    Load images. ground truth labels and masks from the provided dataset source.
    With an inferred label class for the background (assumed to be not provided in the input)

    :param dataset_source: The dataset source for which channels are to be loaded into memory.
    :return: a Sample object with the loaded volume (image), labels, mask and metadata.
    """
    images = [load_nifti_image(channel, ImageDataType.IMAGE.value) for channel in dataset_source.image_channels]
    image = np.stack([image.image for image in images])

    mask = np.ones_like(image[0], ImageDataType.MASK.value) if dataset_source.mask_channel is None \
        else load_nifti_image(dataset_source.mask_channel, ImageDataType.MASK.value).image

    # create raw sample to return
    metadata = copy(dataset_source.metadata)
    # noinspection PyTypeHints
    metadata.image_header = images[0].header
    return Sample(image=image,
                  labels=load_labels_from_dataset_source(dataset_source),
                  mask=mask,
                  metadata=metadata)


def store_image_as_short_nifti(image: np.ndarray,
                               header: ImageHeader,
                               file_name: str,
                               args: Optional[SegmentationModelBase]) -> str:
    """
    Saves an image in nifti format as ubyte, and performs the following operations:
    1) transpose the image back into X,Y,Z from Z,Y,X
    2) perform linear scaling from config.output_range to window level range, or
    byte range (0,255) if norm_method is not CT Window and scale is True
    3) cast the image values to ubyte before saving

    :param image: 3D image in shape: Z x Y x X.
    :param header: ImageHeader
    :param file_name: The name of the file for this image.
    :param args: The model config.
    :return: the path to the saved image
    """
    if args is not None and args.norm_method == PhotometricNormalizationMethod.CtWindow:
        output_range = get_range_for_window_level(args.level, args.window)
        return store_as_nifti(image=image, header=header, file_name=file_name, image_type=np.short,
                              scale=True, input_range=args.output_range, output_range=output_range)
    # All normalization apart from CT Window can't be easily undone. The image should be somewhere in the
    # range -1 to 1, scale up by a factor of 1000 so that we can use all of the np.short range.
    return store_as_nifti(image=image * 1000, header=header, file_name=file_name, image_type=np.short)


def store_posteriors_as_nifti(image: np.ndarray, header: ImageHeader, file_name: str) -> str:
    """
    Saves an array of posteriors in nifti format as ubyte, and performs the following operations:
    1) transpose the image back into X,Y,Z from Z,Y,X
    2) perform a linear scaling from [0, 1] to byte range
    3) cast the image values to ubyte before saving

    :param image: 3D image in shape: Z x Y x X.
    :param header: Image header for the image
    :param file_name: The name of the file for this image.
    :return: the path to the saved image
    """
    check_array_range(image, DEFAULT_POSTERIOR_VALUE_RANGE, error_prefix="Posterior")
    return store_as_scaled_ubyte_nifti(image=image,
                                       header=header,
                                       file_name=file_name,
                                       input_range=DEFAULT_POSTERIOR_VALUE_RANGE)


def store_as_scaled_ubyte_nifti(image: np.ndarray,
                                header: ImageHeader,
                                file_name: str,
                                input_range: Union[Iterable[int], Iterable[float]]) -> str:
    """
    Saves an image in nifti format as ubyte, and performs the following operations:
    1) transpose the image back into X,Y,Z from Z,Y,X
    2) perform linear scaling from input range to byte range
    3) cast the image values to ubyte before saving

    :param image: 3D image in shape: Z x Y x X.
    :param header: The image header
    :param file_name: The name of the file for this image.
    :param input_range: The input range the image belongs to.
    :return: the path to the saved image
    """
    if input_range is None:
        raise Exception("Input range must be provided")

    ubyte_range = [np.iinfo(np.ubyte).min, np.iinfo(np.ubyte).max]
    return store_as_nifti(image, header, file_name, np.ubyte, True, input_range=input_range,
                          output_range=ubyte_range)


def store_as_ubyte_nifti(image: np.ndarray,
                         header: ImageHeader,
                         file_name: str) -> str:
    """
    Saves an image in nifti format as ubyte, and performs the following operations:
    1) transpose the image back into X,Y,Z from Z,Y,X
    2) cast the image values to ubyte before saving

    :param image: 3D image in shape: Z x Y x X.
    :param header: The image spacing Z x Y x X
    :param file_name: The name of the file for this image.
    :return: the path to the saved image
    """
    return store_as_nifti(image, header, file_name, np.ubyte)


def store_binary_mask_as_nifti(image: np.ndarray, header: ImageHeader, file_name: str) -> str:
    """
    Saves a binary mask to nifti format, and performs the following operations:
    1) Check that the image really only contains binary values (0 and 1)
    2) transpose the image back into X,Y,Z from Z,Y,X
    3) cast the image values to ubyte before saving

    :param image: binary 3D image in shape: Z x Y x X.
    :param header: The image header
    :param file_name: The name of the file for this image.
    :return: the path to the saved image
    :raises: when image is not binary
    """
    if not is_binary_array(image):
        raise Exception("Array values must be binary.")

    return store_as_nifti(image=image, header=header, file_name=file_name, image_type=np.ubyte)


def store_as_nifti(image: np.ndarray,
                   header: ImageHeader,
                   file_name: str,
                   image_type: Union[str, type, np.dtype],
                   scale: bool = False,
                   input_range: Optional[Iterable[Union[int, float]]] = None,
                   output_range: Optional[Iterable[Union[int, float]]] = None) -> str:
    """
    Saves an image in nifti format (uploading to Azure also if an online Run), and performs the following operations:
    1) transpose the image back into X,Y,Z from Z,Y,X
    2) if scale is true, then performs a linear scaling to either the input/output range or
    byte range (0,255) as default.
    3) cast the image values to the given type before saving

    :param image: 3D image in shape: Z x Y x X.
    :param header: The image header
    :param file_name: The name of the file for this image.
    :param scale: Should perform linear scaling of the image, to desired output range or byte range by default.
    :param input_range: The input range the image belongs to.
    :param output_range: The output range to scale the image to.
    :param image_type: The type to save the image in.
    :return: the path to the saved image
    """
    if image.ndim != 3:
        raise Exception("Image must have 3 dimensions, found: {}".format(len(image.shape)))

    if image_type is None:
        raise Exception("You must specify a valid image type.")

    if scale and ((input_range is not None and output_range is None)
                  or (input_range is None and output_range is not None)):
        raise Exception("You must provide both input and output ranges to apply custom linear scaling.")

    # rescale image for visualization in the app
    if scale:
        if input_range is not None and output_range is not None:
            # noinspection PyTypeChecker
            image = LinearTransform.transform(
                data=image,
                input_range=input_range,  # type: ignore
                output_range=tuple(output_range)  # type: ignore
            )
        else:
            image = (image + 1) * 255

    image = sitk.GetImageFromArray(image.astype(image_type))
    image.SetSpacing(sitk.VectorDouble(reverse_tuple_float3(header.spacing)))  # Spacing needs to be X Y Z
    image.SetOrigin(header.origin)
    image.SetDirection(header.direction)
    sitk.WriteImage(image, file_name)
    return file_name


def save_lines_to_file(file: Path, values: List[str]) -> None:
    """
    Writes an array of lines into a file, one value per line. End of line character is hardcoded to be `\n`.
    If the file exists already, it will be deleted.

    :param file: The path where to save the file
    :param values: A list of strings
    """
    if file.exists():
        file.unlink()

    lines = map(lambda l: l + "\n", values)
    file.write_text("".join(lines))


def reverse_tuple_float3(tuple: TupleFloat3) -> TupleFloat3:
    """
    Reverse a tuple of 3 floats.

    :param tuple: of 3 floats
    :return: a tuple of 3 floats reversed
    """
    return tuple[2], tuple[1], tuple[0]


def tabulate_dataframe(df: pd.DataFrame, pefix_newline: bool = True) -> str:
    """
    Helper function to print a pandas Dataframe in a nicely readable table.
    """
    return ("\n" if pefix_newline else "") + tabulate(df, tablefmt="fancy_grid", headers="keys", showindex="never")

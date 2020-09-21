#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import param
from azureml._restclient.constants import RunStatus
from azureml.core import Run
from scipy.ndimage.morphology import binary_erosion, distance_transform_edt, generate_binary_structure

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import fetch_child_runs, fetch_run
from InnerEye.Common.common_util import FULL_METRICS_DATAFRAME_FILE, epoch_folder_name
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import load_nifti_image


class SurfaceDistanceRunType(Enum):
    IOV = 'IOV'
    OUTLIERS = 'OUTLIERS'


class Plane(Enum):
    SAGITTAL = 'SAGITTAL'
    CORONAL = 'CORONAL'
    AXIAL = 'AXIAL'


class SurfaceDistanceConfig(GenericConfig):
    run_mode: SurfaceDistanceRunType = param.ClassSelector(default=SurfaceDistanceRunType.IOV,
                                                           class_=SurfaceDistanceRunType,
                                                           doc="Type of study: either IOV or OUTLIERS")
    annotators: List[str] = param.List(class_=str, default=[],
                                       doc="Names/initials of annotators as displayed in filenames",
                                       instantiate=False)
    model_name: str = param.String(None, doc="The name of the model as displayed in filenames")
    plane: Plane = param.ClassSelector(default=Plane.SAGITTAL,
                                       class_=Plane,
                                       instantiate=False,
                                       doc="The plane to show images in. Default is sagittal, can also be axial or "
                                           "coronal")
    execution_mode: ModelExecutionMode = param.ClassSelector(default=ModelExecutionMode.TEST, class_=ModelExecutionMode,
                                                             doc="The model execution mode we are plotting images for."
                                                                 " Default is Test, can also be Train or Val")
    outlier_range: float = param.Number(1.0, doc="Number of standard deviations away from the mean to "
                                                 "use for outlier range")
    run_recovery_id: str = param.String(None, doc="The recovery id of the run to download from.")
    ground_truth_dir: str = param.String(None, doc="Name of the dir in which to find ground truth data")
    output_img_dir: str = param.String(None, doc="The name of the directory in which to store plots")


def get_first_child_run(azure_config: AzureConfig) -> Run:
    """
    Download first child run in order to download data
    :param azure_config:
    :return: first child run
    """
    if not azure_config.run_recovery_id:
        raise ValueError("azure_config.run_recovery_id is not provided.")
    workspace = azure_config.get_workspace()
    hyperdrive_run = fetch_run(workspace, azure_config.run_recovery_id)
    child_runs = fetch_child_runs(hyperdrive_run, status=RunStatus.COMPLETED)
    return child_runs[0]


def load_ground_truth_from_run(model_config: SegmentationModelBase, sd_config: SurfaceDistanceConfig, subject_id: int,
                               structure: str) -> np.ndarray:
    """
    For outliers, load individual ground truth file for a given dataset, subject ID and structure name
    :param model_config:
    :param sd_config:
    :param subject_id: ID of the given subject
    :param structure: Name of the anatomical structure
    :return: ground truth array
    """
    ground_truth_path = model_config.outputs_folder / sd_config.run_recovery_id / sd_config.ground_truth_dir \
                        / str(subject_id) / f"{structure}.nii.gz"
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"No file exists at {ground_truth_path}")
    image = io_util.load_nifti_image(ground_truth_path).image
    return image


def get_run_output_dir(azure_config: AzureConfig, model_config: SegmentationModelBase) -> Path:
    """
    Get the directory where Azure run's output data will be stored. the total filepath will depend on which
    container we download data from.
    :param azure_config:
    :param model_config:
    :return output_dir: directory that all artifact paths use as a prefix
    """
    if not azure_config.run_recovery_id:
        raise ValueError("azure_config.run_recovery_id is not provided")

    run_recovery_id = azure_config.run_recovery_id
    output_dir = model_config.outputs_folder / run_recovery_id
    return output_dir


def dir_for_subject(azure_config: AzureConfig, model_config: SegmentationModelBase, prefix: Path) -> Path:
    """
    Combine the local data dir and the Azure dir we are downloading images
    from to get the directory the images for a subject will be downlaoded to
    :param azure_config: AzureConfig
    :param model_config: Config
    :param prefix:
    :return:
    """
    src = get_run_output_dir(azure_config, model_config)
    subject_dir = src / prefix
    if not subject_dir.is_dir():
        raise NotADirectoryError(f"No directory exists at {subject_dir}")
    return subject_dir


def get_metrics_path(azure_config: AzureConfig, model_config: SegmentationModelBase) -> Path:
    """
    Get path to metrics.csv file for a downlaoded run, for the purpose of determining outliers
    :param azure_config: AzureConfig
    :param model_config: Config
    :return:
    """
    src = get_run_output_dir(azure_config, model_config)
    num_epochs_dir = epoch_folder_name(model_config.num_epochs)
    root = src / num_epochs_dir
    if not root.is_dir():
        raise NotADirectoryError(f"Dir doesnt exist: {root}")

    metrics_path = root / FULL_METRICS_DATAFRAME_FILE
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics path does not exist at location {metrics_path}")
    return metrics_path


def get_subject_prefix(model_config: SegmentationModelBase, train_mode: ModelExecutionMode, subject_id: int) -> Path:
    """
    Returns the path to subject dir for a given model and train mode
    :param model_config: Config
    :param train_mode: Model execution mode -i.e. train, test or val
    :param subject_id: ID of the subject
    :return prefix: the filepath prefix within the container from which to download all artifacts
    """
    num_epochs_dir = epoch_folder_name(model_config.num_epochs)
    prefix = model_config.outputs_folder / num_epochs_dir / train_mode.value / "{0:03d}".format(subject_id)
    return prefix


def initialise_surface_distance_dictionary(annotators: List[str], arr_shape: Tuple[int, int, int]
                                           ) -> Dict[str, np.ndarray]:
    """
    Given a list of annotators and the image size expected for all surface distance plots,
    return a dictionary where keys are annotators and values are zeros in shape of entire image,
    so that surface distances for each structure can be added to one plot.
    :param annotators: List of the annotator names as they appear in filepaths
    :param arr_shape: Shape of the array to be intialized
    :return:
    """
    return {a: np.zeros(arr_shape) for a in annotators}


def get_majority_vote(arr_list: List[np.ndarray]) -> np.ndarray:
    """
    Given a list of label arrays, get the majority vote at each voxel
    :param arr_list:
    :return:
    """
    arr = np.array(arr_list)
    half_voters = len(arr) / 2 if len(arr) % 2 == 0 else (len(arr) + 1) / 2
    majority_vote = np.array((arr == 1).sum(axis=0) >= half_voters).astype(int)
    return majority_vote


def get_annotations_and_majority_vote(model_config: SegmentationModelBase, annotators: List[str], structure_name: str
                                      ) -> np.ndarray:
    """
    Load each annotation and calculate the 'gold standard' segmentation (with majority voting)
    :param model_config: Config
    :param annotators: List of the annotator names as they appear in filepaths
    :param structure_name: Name of the anatomical structure
    :return:
    """
    iov_dir = model_config.outputs_folder / "iov"
    segmentations = []
    logging.info(f"Annotators going into gold standard: {annotators}")
    for annotator_num, annotator in enumerate(annotators):
        segmentation_path = iov_dir / f"{structure_name}{annotator}.nii.gz"
        segmentation = load_nifti_image(segmentation_path).image
        segmentations.append(segmentation)

    majority_vote_seg = get_majority_vote(segmentations)
    return majority_vote_seg


def extract_border(img: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Get contour by calculating eroded version of the image and subtracting from the original.
    :param img: Array containing structure from which to extract the border
    :param connectivity: integer determining which pixels are considered neighbours of the central element,
    ranging from 1 = no diagonal elements and rank = all elements
    :return:
    """
    if not np.unique(img.astype(int)).tolist() == [0, 1]:
        raise ValueError("In order to extract border, you must provide a binary image")
    # binary structure (kernel for detecting edges)
    structuring_element = generate_binary_structure(img.ndim, connectivity)
    erosion = binary_erosion(img, structure=structuring_element, iterations=1)
    # subtract eroded image from original
    img_border = img - erosion
    return img_border


def calculate_surface_distances(ground_truth: np.ndarray, pred: np.ndarray, voxel_spacing: Union[float, List[float]]
                                ) -> np.ndarray:
    """
    Calculate the Euclidean surface distance between a given prediction and the 'ground truth'
    :param ground_truth: 3D binary array (X x Y x Z) of ground truth segmentation
    :param pred: 3D binary array (X x Y x Z) of predicted segmentation
    :param voxel_spacing: voxel spacing, taken from the image header (transposed if applicable)
    :return:
    """
    gt_contour = extract_border(ground_truth, connectivity=3)
    pred_contour = extract_border(pred, connectivity=3)

    inverted_gt = np.logical_not(gt_contour)
    inverted_pred = np.logical_not(pred_contour)

    # calculate euclidean distances (invert contour to give surface pixels distance value of 0)
    edt_ground_truth = distance_transform_edt(inverted_gt, sampling=voxel_spacing)
    edt_pred = distance_transform_edt(inverted_pred, sampling=voxel_spacing)

    # Only interested in parts where ground truth and prediction disagree
    sds_full_mask = np.array(pred != ground_truth)
    sds_pred_to_gt = np.multiply(edt_ground_truth, sds_full_mask)
    sds_gt_to_pred = np.multiply(edt_pred, sds_full_mask)

    sds_full = np.mean([np.abs(sds_pred_to_gt), np.abs(sds_gt_to_pred)], axis=0)

    if np.count_nonzero(sds_full) == 0:
        logging.warning("No non-zero surface distances")

    return sds_full

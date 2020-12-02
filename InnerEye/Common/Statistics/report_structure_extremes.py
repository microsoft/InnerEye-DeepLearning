#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Script to generate a report on the structures in a dataset. The single argument should be a dataset
directory, containing subject subdirectories each with a numerical name, and in turn containing a "*.nii.gz"
file for each structure.

The output is a table with header

xLo xHi xMx yLo yHi yMx zLo zHi zMx    Sub Series   Structure    Missing

where xLo and xHi are the lowest and highest x (sagittal) values of any voxel in the structure, and xMx is the maximum
x value in the image, i.e. the size in the x direction minus one. Similarly for the columns headed "y" (coronal) and
"z" (axial). "Sub" and "Structure" are the subject and structure being analyzed (i.e. the file is Sub/Structure.nii.gz).
If a structure is empty, all the "Lo" and "Hi" values are given as -1.

If any slices (in any dimension) are missing, they are indicated like this after the file name:

 91 187 500 140 276 500   0 108 139    258 femur_r         zMs:79-107

This means: slices 79 to 107 inclusive in the z direction are missing, i.e. there is at least one slice below
79 that contains voxels for the structure, and at least one slice above 107, but none in the range 79 to 107.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, TextIO, Tuple

import numpy as np
import param
from azure.storage.blob.blockblobservice import BlockBlobService

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.utils.blobxfer_util import download_blobs
from InnerEye.ML.utils.io_util import read_image_as_array_with_header

MISSING_SLICE_MARKER = "Ms:"
NIFTI_SUFFIX = ".nii.gz"


class ReportStructureExtremesConfig(GenericConfig):
    dataset: str = param.String(default=".", doc="Dataset name or directory to process")
    settings: Path = param.ClassSelector(class_=Path, default=fixed_paths.SETTINGS_YAML_FILE,
                                         doc="YAML file that contains settings to access Azure.")


def populate_series_maps(dataset_csv_path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    series_map = {}
    institution_map = {}
    with open(dataset_csv_path) as file:
        for row in csv.reader(file):
            series_map[row[0]] = row[3]
            institution_map[row[0]] = row[4]
    return series_map, institution_map


def open_with_header(path: str, path_set: Set[str]) -> TextIO:
    exists = os.path.isfile(path)
    file = open(path, "a")
    if not exists:
        path_set.add(path)
        file.write("xLo xHi xMx yLo yHi yMx zLo zHi zMx    Sub Series   Structure       Missing\n")
    return file


def report_structure_extremes(dataset_dir: str, azure_config: AzureConfig) -> None:
    """
    Writes structure-extreme lines for the subjects in a directory.
    If there are any structures with missing slices, a ValueError is raised after writing all the lines.
    This allows a build failure to be triggered when such structures exist.
    :param azure_config: An object with all necessary information for accessing Azure.
    :param dataset_dir: directory containing subject subdirectories with integer names.
    """
    download_dataset_directory(azure_config, dataset_dir)
    subjects: Set[int] = set()
    series_map = None
    institution_map = None
    for subj in os.listdir(dataset_dir):
        try:
            subjects.add(int(subj))
        except ValueError:
            if subj == "dataset.csv":
                # We should find this in every dataset_dir.
                series_map, institution_map = populate_series_maps(os.path.join(dataset_dir, subj))
            pass
    if institution_map is None or series_map is None:
        raise FileNotFoundError(f"Cannot find {dataset_dir}/dataset.csv")
    if not subjects:
        print(f"No subject directories found in {dataset_dir}")
        return
    print(f"Found {len(subjects)} subjects in {dataset_dir}")
    # You could temporarily edit subjects to be an explicit list of integers here, to process only certain subjects:
    # subjects = [23, 42, 99]
    full_output_dir = os.path.join(dataset_dir, "structure_extremes_full")
    os.makedirs(full_output_dir)
    problems_output_dir = os.path.join(dataset_dir, "structure_extremes_problems")
    os.makedirs(problems_output_dir)
    n_missing = 0
    files_created: Set[str] = set()
    for (index, subj_int) in enumerate(sorted(subjects), 1):
        subj = str(subj_int)
        institution_id = institution_map.get(subj, "")
        out = open_with_header(os.path.join(full_output_dir, institution_id + ".txt"), files_created)
        err = None
        for line in report_structure_extremes_for_subject(os.path.join(dataset_dir, subj), series_map[subj]):
            out.write(line + "\n")
            if line.find(MISSING_SLICE_MARKER) > 0:
                if err is None:
                    err = open_with_header(os.path.join(problems_output_dir, institution_id + ".txt"), files_created)
                err.write(line + "\n")
                n_missing += 1
        out.close()
        if err is not None:
            err.close()
        if index % 25 == 0:
            print(f"Processed {index} subjects")
    print(f"Processed all {len(subjects)} subjects")
    upload_to_dataset_directory(azure_config, dataset_dir, files_created)
    # If we found any structures with missing slices, raise an exception, which should be
    # uncaught where necessary to make any appropriate build step fail.
    if n_missing > 0:
        raise ValueError(f"Found {n_missing} structures with missing slices")


def download_dataset_directory(azure_config: AzureConfig, dataset_dir: str) -> bool:
    if os.path.isdir(dataset_dir):
        return False
    account_key = azure_config.get_dataset_storage_account_key()
    blobs_root_path = os.path.join(azure_config.datasets_container, os.path.basename(dataset_dir)) + "/"
    sys.stdout.write(f"Downloading data to {dataset_dir} ...")
    assert account_key is not None  # for mypy
    download_blobs(azure_config.datasets_storage_account, account_key, blobs_root_path, Path(dataset_dir))
    sys.stdout.write("done\n")
    return True


def upload_to_dataset_directory(azure_config: AzureConfig, dataset_dir: str, files: Set[str]) -> None:
    if not files:
        return
    account_key = azure_config.get_dataset_storage_account_key()
    block_blob_service = BlockBlobService(account_name=azure_config.datasets_storage_account, account_key=account_key)
    container_name = os.path.join(azure_config.datasets_container, os.path.basename(dataset_dir))
    for path in files:
        blob_name = path[len(dataset_dir) + 1:]
        block_blob_service.create_blob_from_path(container_name, blob_name, path)
        print(f"Uploaded {path} to {azure_config.datasets_storage_account}:{container_name}/{blob_name}")


def report_structure_extremes_for_subject(subj_dir: str, series_id: str) -> Iterator[str]:
    """
    :param subj_dir: subject directory, containing <structure>.nii.gz files
    :param series_id: series identifier for the subject
    Yields a line for every <structure>.nii.gz file in the directory.
    """
    subject = os.path.basename(subj_dir)
    series_prefix = "" if series_id is None else series_id[:8]
    for base in sorted(os.listdir(subj_dir)):
        # We ignore ct.nii.gz files; we should ignore other image files too.
        if base != "ct" + NIFTI_SUFFIX and base.endswith(NIFTI_SUFFIX):
            data, _ = read_image_as_array_with_header(Path(os.path.join(subj_dir, base)))
            yield line_for_structure(subject, series_prefix, base, data)


def line_for_structure(subject: str, series_prefix: str, base: str, data: np.array) -> str:
    """
    :param subject: a subject, to include in the result
    :param series_prefix: first 8 characters (if any) of the series ID of the subject
    :param base: a file basename ending in ".nii.gz", to include in the result
    :param data: an array of 0's and 1's delineating the structure
    :return: a line to add to the report
    """
    # Array of x values where some voxel is positive.
    presence_x = np.where((data > 0).any(axis=1).any(axis=1))[0]
    # Array of y values where some voxel is positive.
    presence_y = np.where((data > 0).any(axis=0).any(axis=1))[0]
    # Array of z values where some voxel is positive.
    presence_z = np.where((data > 0).any(axis=0).any(axis=0))[0]
    shape = data.shape
    values = [extent_list(presence_x, shape[0] - 1),
              extent_list(presence_y, shape[1] - 1),
              extent_list(presence_z, shape[2] - 1)]
    line = ""
    suffix = ""
    for xyz, pairs in zip(("x", "y", "z"), values):
        line += " ".join(f"{value:3d}" for value in pairs[0]) + " "
        if pairs[1]:
            suffix += f" {xyz}{MISSING_SLICE_MARKER}" + ",".join(pairs[1])
    structure = base[:-len(NIFTI_SUFFIX)]
    line += f"   {subject:>3s} {series_prefix:<8s} {structure:<15s}{suffix}"
    return line


def extent_list(presence: np.array, max_value: int) -> Tuple[List[int], List[str]]:
    """
    :param presence: a 1-D array of distinct integers in increasing order.
    :param max_value: any integer, not necessarily related to presence
    :return: two tuples: (1) a list of the minimum and maximum values of presence, and max_value;
    (2) a list of strings, each denoting a missing range of values within "presence".
    """
    if len(presence) == 0:
        return [-1, -1, max_value], []
    n_missing = presence[-1] - presence[0] + 1 - len(presence)
    result = [presence[0], presence[-1], max_value]
    if n_missing == 0:
        return result, []
    missing_ranges = derive_missing_ranges(presence)
    return result, missing_ranges


def derive_missing_ranges(presence: np.array) -> List[str]:
    """
    :param presence: a 1-D array of distinct integers in increasing order.
    :return: a list of strings, each denoting a missing range of values within "presence".
    """
    # "missing" is a list of pairs, each of which is [start, end] of most recent
    # range found so far.
    missing = []
    previous = presence[0]
    for current in presence[1:]:
        if current != previous + 1:
            missing.append((previous + 1, current - 1))
        previous = current
    missing_ranges = [f"{i}" if i == j else f"{i}-{j}" for i, j in missing]
    return missing_ranges


def main(settings_yaml_file: Optional[Path] = None,
         project_root: Optional[Path] = None) -> None:
    """
    Main function.
    """
    logging_to_stdout()
    config = ReportStructureExtremesConfig.parse_args()
    azure_config = AzureConfig.from_yaml(yaml_file_path=settings_yaml_file or config.settings,
                                         project_root=project_root)
    report_structure_extremes(config.dataset, azure_config)


if __name__ == '__main__':
    main(project_root=fixed_paths.repository_root_directory())

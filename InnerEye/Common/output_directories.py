#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import shutil
from dataclasses import dataclass

from InnerEye.Common import common_util
from InnerEye.Common.type_annotations import PathOrString


def make_test_output_dir(folder: PathOrString) -> None:
    """
    Delete the folder if it exists, and remakes it. This method ignores errors that can come from
    an explorer window still being open inside of the test result folder.
    """
    folder = str(folder)

    if os.path.exists(folder):
        shutil.rmtree(folder, ignore_errors=True)

    os.makedirs(folder, exist_ok=True)


@dataclass(frozen=True)
class TestOutputDirectories:
    """
    Data class for the output directories for a given test
    """
    root_dir: str

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

    def create_file_or_folder_path(self, file_or_folder_name: str) -> str:
        """
        Prepends root dir to the given file or folder name
        :param file_or_folder_name: Name of file or folder to be created under root_dir
        """
        return os.path.join(self.root_dir, file_or_folder_name)

    def make_sub_dir(self, dir_name: str) -> str:
        """
        Makes a sub directory under root_dir
        :param dir_name: Name of subdirectory to be created.
        """
        sub_dir_path = os.path.join(self.root_dir, dir_name)
        os.makedirs(sub_dir_path)
        return str(sub_dir_path)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import shutil
from dataclasses import dataclass
from pathlib import Path

from InnerEye.Common import common_util
from InnerEye.Common.type_annotations import PathOrString


def remove_and_create_folder(folder: PathOrString) -> None:
    """
    Delete the folder if it exists, and remakes it. This method ignores errors that can come from
    an explorer window still being open inside of the test result folder.
    """
    folder = Path(folder)

    if folder.is_dir():
        shutil.rmtree(folder, ignore_errors=True)

    folder.mkdir(exist_ok=True)


@dataclass(frozen=True)
class OutputFolderForTests:
    """
    Data class for the output directories for a given test
    """
    root_dir: Path

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

    def create_file_or_folder_path(self, file_or_folder_name: str) -> Path:
        """
        Creates a full path for the given file or folder name relative to the root directory stored in the present
        object.
        :param file_or_folder_name: Name of file or folder to be created under root_dir
        """
        return self.root_dir / file_or_folder_name

    def make_sub_dir(self, dir_name: str) -> Path:
        """
        Makes a sub directory under root_dir
        :param dir_name: Name of subdirectory to be created.
        """
        sub_dir_path = self.create_file_or_folder_path(dir_name)
        sub_dir_path.mkdir()
        return sub_dir_path

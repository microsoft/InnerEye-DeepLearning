#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from pathlib import Path

from health_azure import download_files_from_run_id, get_workspace
from InnerEye.Common import fixed_paths


def download_file_if_necessary(run_id: str, remote_dir: Path, download_dir: Path, filename: str) -> None:
    """
    Function to download any file from an AML run if it doesn't exist locally
    :param run_id: run ID of the AML run
    :param remote_dir: remote directory from where the file is downloaded
    :param download_dir: local directory where to save the downloaded file
    :param filename: name of the file to be downloaded (e.g. `"test_output.csv"`).
    """
    aml_workspace = get_workspace()
    # current is the config file level
    current = Path(__file__)
    # os.chdir(fixed_paths.repository_root_directory())
    os.chdir(current.parent.parent.parent)
    local_path = download_dir / run_id.split(":")[1] / "outputs" / filename
    remote_path = remote_dir / filename
    if local_path.exists():
        print("File already exists at", local_path)
    else:
        local_dir = local_path.parent.parent
        local_dir.mkdir(exist_ok=True, parents=True)
        download_files_from_run_id(run_id=run_id,
                               output_folder=local_dir,
                               prefix=str(remote_path),
                               aml_workspace=aml_workspace,
                               validate_checksum=True)
        assert local_path.exists()
        print("File is downloaded at", local_path)

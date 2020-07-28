#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
from pathlib import Path
from typing import Optional

from InnerEye.Azure.azure_config import AzureConfig, ExperimentResultLocation

BUILDINFORMATION_JSON = "buildinformation.json"


def build_information_to_dot_net_json(azure_config: AzureConfig, result_location: ExperimentResultLocation) -> str:
    """
    Converts the build metadata to a JSON string.
    :param azure_config: Azure configuration file with build information.
    :param result_location: ExperimentResultLocation object with result locations.
    """
    return json.dumps({
        "BuildNumber": azure_config.build_number,
        "BuildRequestedFor": azure_config.build_user,
        "BuildSourceBranchName": azure_config.build_branch,
        "BuildSourceVersion": azure_config.build_source_id,
        "BuildSourceAuthor": azure_config.build_source_author,
        "ModelName": azure_config.model,
        "ResultsContainerName": result_location.results_container_name,
        "ResultsUri": result_location.results_uri,
        "DatasetFolder": result_location.dataset_folder,
        "DatasetFolderUri": result_location.dataset_uri,
        "AzureBatchJobName": result_location.azure_job_name})


def build_information_to_dot_net_json_file(azure_config: AzureConfig,
                                           result_location: ExperimentResultLocation,
                                           folder: Optional[Path] = None) -> None:
    """
    Writes the build metadata to a file called buildinformation.json in the given folder.
    :param azure_config: Azure configuration file
    :param result_location: ExperimentResultLocation object with result locations.
    :param folder: Results are written to this folder, if not None. Else, results are written in the root folder.
    """
    filename = Path(BUILDINFORMATION_JSON)

    if folder is not None:
        if not folder.exists():
            folder.mkdir(parents=True)

    full_file = filename if folder is None else folder / filename
    with full_file.open("w") as f:
        f.write(build_information_to_dot_net_json(azure_config, result_location))

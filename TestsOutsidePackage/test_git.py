#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths


def test_git_info() -> None:
    """
    Test if git branch information can be read correctly.
    """
    azure_config = AzureConfig.from_yaml(fixed_paths.TRAIN_YAML_FILE)
    azure_config.project_root = fixed_paths.repository_root_directory()
    assert azure_config.build_branch == ""
    assert azure_config.build_source_id == ""
    assert azure_config.build_source_author == ""
    assert azure_config.build_source_message == ""
    assert azure_config.build_source_repository == ""
    source_info = azure_config.get_git_information()
    assert source_info.repository == azure_config.project_root.name
    assert len(source_info.branch) > 0
    assert len(source_info.commit_id) == 40
    assert len(source_info.commit_author) > 0
    assert len(source_info.commit_message) > 0


def test_git_info_from_commandline() -> None:
    """
    Test if git branch information can be overriden on the commandline
    """
    azure_config = AzureConfig.from_yaml(fixed_paths.TRAIN_YAML_FILE)
    azure_config.project_root = fixed_paths.repository_root_directory()
    azure_config.build_branch = "branch"
    azure_config.build_source_id = "id"
    azure_config.build_source_author = "author"
    azure_config.build_source_message = "message"
    azure_config.build_source_repository = "repo"
    source_info = azure_config.get_git_information()
    assert source_info.branch == "branch"
    assert source_info.commit_id == "id"
    assert source_info.commit_author == "author"
    assert source_info.commit_message == "message"
    assert source_info.repository == "repo"

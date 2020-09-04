#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import fixed_paths

# Re-compute the repository root directory, because we need to have that point to the git repo, not the root
# of the InnerEye package
from InnerEye.Common.common_util import logging_to_stdout

project_root = Path(__file__).resolve().parent.parent.parent


def test_git_info() -> None:
    """
    Test if git branch information can be read correctly.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    azure_config = AzureConfig.from_yaml(fixed_paths.TRAIN_YAML_FILE)
    azure_config.project_root = project_root
    assert azure_config.build_branch == ""
    assert azure_config.build_source_id == ""
    assert azure_config.build_source_author == ""
    assert azure_config.build_source_message == ""
    assert azure_config.build_source_repository == ""
    source_info = azure_config.get_git_information()
    assert source_info.repository == azure_config.project_root.name
    # We can't access the branch name when this test runs on the build agents, because the repositories
    # are checked out in "detached head" state. Hence, can't assert on branch name in any way
    # that works locally and in the cloud.
    assert len(source_info.commit_id) == 40
    assert len(source_info.commit_author) > 0
    assert len(source_info.commit_message) > 0


def test_git_info_from_commandline() -> None:
    """
    Test if git branch information can be overriden on the commandline
    """
    azure_config = AzureConfig.from_yaml(fixed_paths.TRAIN_YAML_FILE)
    azure_config.project_root = project_root
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

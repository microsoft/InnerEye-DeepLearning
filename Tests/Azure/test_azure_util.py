#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from azureml.core import Run
from azureml.core.workspace import Workspace
from azureml.core.conda_dependencies import CondaDependencies

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import create_experiment_name, pytorch_version_from_conda_dependencies
from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, fetch_child_runs, fetch_run, \
    get_cross_validation_split_index, is_cross_validation_child_run, is_run_and_child_runs_completed, \
    merge_conda_dependencies, \
    merge_conda_files, to_azure_friendly_container_path
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import ENVIRONMENT_YAML_FILE_NAME
from InnerEye.Common.output_directories import OutputFolderForTests
from Tests.ML.util import get_default_workspace
from Tests.AfterTraining.test_after_training import get_most_recent_run


def test_os_path_to_azure_friendly_container_path() -> None:
    """
    Check paths correctly converted to single forward slash formats expected by Azure.
    """
    assert "a" == to_azure_friendly_container_path(Path("a"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a/b/c"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a//b//c"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a\\b/c/"))


@pytest.mark.after_training_single_run
def test_get_cross_validation_split_index_single_run() -> None:
    """
    Test that retrieved cross validation split index is as expected, for single runs.
    """
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=get_most_recent_run()
    )
    # check for offline run
    assert get_cross_validation_split_index(Run.get_context()) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    # check for online runs
    assert get_cross_validation_split_index(run) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX


@pytest.mark.after_training_ensemble_run
def test_get_cross_validation_split_index_ensemble_run() -> None:
    """
    Test that retrieved cross validation split index is as expected, for ensembles.
    """
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=get_most_recent_run()
    )
    # check for offline run
    assert get_cross_validation_split_index(Run.get_context()) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    # check for online runs
    assert get_cross_validation_split_index(run) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    assert all([get_cross_validation_split_index(x) > DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
                for x in fetch_child_runs(run)])


@pytest.mark.after_training_single_run
def test_is_cross_validation_child_run_single_run() -> None:
    """
    Test that cross validation child runs are identified correctly. A single run should not be identified as a
    cross validation run.
    """
    rid = get_most_recent_run()
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=rid
    )
    # check for offline run
    assert not is_cross_validation_child_run(Run.get_context())
    # check for online runs
    assert not is_cross_validation_child_run(run)


@pytest.mark.after_training_ensemble_run
def test_is_cross_validation_child_run_ensemble_run() -> None:
    """
    Test that cross validation child runs are identified correctly.
    """
    rid = get_most_recent_run()
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=rid
    )
    # check for offline run
    assert not is_cross_validation_child_run(Run.get_context())
    # check for online runs
    assert not is_cross_validation_child_run(run)
    assert all([is_cross_validation_child_run(x) for x in fetch_child_runs(run)])


def test_merge_conda(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests the logic for merging Conda environment files.
    """
    env1 = """
channels:
  - defaults
  - pytorch
dependencies:
  - conda1=1.0
  - conda2=2.0
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.7.0
      - foo==1.0
"""
    env2 = """
channels:
  - defaults
dependencies:
  - conda1=1.1
  - conda_both=3.0
  - pip:
      - azureml-sdk==1.6.0
      - bar==2.0
"""
    file1 = test_output_dirs.root_dir / "env1.yml"
    file1.write_text(env1)
    file2 = test_output_dirs.root_dir / "env2.yml"
    file2.write_text(env2)
    files = [file1, file2]
    merged_file = test_output_dirs.root_dir / "merged.yml"
    merge_conda_files(files, merged_file)
    assert merged_file.read_text().splitlines() == """channels:
- defaults
- pytorch
dependencies:
- conda1=1.0
- conda1=1.1
- conda2=2.0
- conda_both=3.0
- pip:
  - azureml-sdk==1.6.0
  - azureml-sdk==1.7.0
  - bar==2.0
  - foo==1.0
""".splitlines()
    conda_dep = merge_conda_dependencies(files)
    # We expect to see the union of channels.
    assert list(conda_dep.conda_channels) == ["defaults", "pytorch"]
    # Package version conflicts are not resolved, both versions are retained.
    assert list(conda_dep.conda_packages) == ["conda1=1.0", "conda1=1.1", "conda2=2.0", "conda_both=3.0"]
    assert list(conda_dep.pip_packages) == ["azureml-sdk==1.6.0", "azureml-sdk==1.7.0", "bar==2.0", "foo==1.0"]


def test_experiment_name() -> None:
    c = AzureConfig()
    c.build_branch = "branch"
    c.get_git_information()
    assert create_experiment_name(c) == "branch"
    c.experiment_name = "foo"
    assert create_experiment_name(c) == "foo"


def test_framework_version(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if the Pytorch framework version can be read correctly from the current environment file.
    """
    environment_file = fixed_paths.repository_root_directory(ENVIRONMENT_YAML_FILE_NAME)
    assert environment_file.is_file(), "Environment file must be present"
    conda_dep = CondaDependencies(conda_dependencies_file_path=environment_file)
    framework = pytorch_version_from_conda_dependencies(conda_dep)
    # If this fails, it is quite likely that the AzureML SDK is behind pytorch, and does not yet know about a
    # new version of pytorch that we are using here.
    assert framework is not None


def get_run_and_check(run_id: str, expected: bool, workspace: Workspace) -> None:
    run = fetch_run(workspace, run_id)
    status = is_run_and_child_runs_completed(run)
    assert status == expected


@pytest.mark.after_training_single_run
def test_is_completed_single_run() -> None:
    """
    Test if we can correctly check run status for a single run.
    :return:
    """
    logging_to_stdout()
    workspace = get_default_workspace()
    get_run_and_check(get_most_recent_run(), True, workspace)


@pytest.mark.after_training_ensemble_run
def test_is_completed_ensemble_run() -> None:
    """
    Test if we can correctly check run status and status of child runs for an ensemble run.
    :return:
    """
    logging_to_stdout()
    workspace = get_default_workspace()
    get_run_and_check(get_most_recent_run(), True, workspace)


# TODO remove reference to hardcoded run ID
def test_is_completed_failed_run() -> None:
    """
    Test if we can correctly check run status and status of child runs for a failed ensemble run.
    :return:
    """
    logging_to_stdout()
    workspace = get_default_workspace()
    # This Hyperdrive run has 1 failing child run, the parent run completed successfully.
    get_run_and_check("refs_pull_326_merge:HD_d123f042-ca58-4e35-9a64-48d71c5f63a7", False, workspace)

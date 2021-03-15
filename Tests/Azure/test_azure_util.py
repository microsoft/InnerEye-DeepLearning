#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import time
from pathlib import Path
from typing import Iterator, Set

import pytest
from azureml.core import Run
from azureml.core.workspace import Workspace

from InnerEye.Azure.azure_config import AzureConfig, SourceConfig
from InnerEye.Azure.azure_runner import create_experiment_name, get_or_create_python_environment
from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, fetch_child_runs, fetch_run, \
    get_cross_validation_split_index, is_cross_validation_child_run, is_run_and_child_runs_completed, \
    merge_conda_dependencies, merge_conda_files, to_azure_friendly_container_path
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import PRIVATE_SETTINGS_FILE, PROJECT_SECRETS_FILE, \
    get_environment_yaml_file, repository_root_directory
from InnerEye.Common.output_directories import OutputFolderForTests
from Tests.AfterTraining.test_after_training import FALLBACK_ENSEMBLE_RUN, get_most_recent_run, get_most_recent_run_id
from Tests.ML.util import get_default_azure_config, get_default_workspace


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
    run = get_most_recent_run()
    # check for offline run
    assert get_cross_validation_split_index(Run.get_context()) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    # check for online runs
    assert get_cross_validation_split_index(run) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX


@pytest.mark.after_training_ensemble_run
def test_get_cross_validation_split_index_ensemble_run() -> None:
    """
    Test that retrieved cross validation split index is as expected, for ensembles.
    """
    # check for offline run
    assert get_cross_validation_split_index(Run.get_context()) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    # check for online runs
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    assert get_cross_validation_split_index(run) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    assert all([get_cross_validation_split_index(x) > DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
                for x in fetch_child_runs(run)])


@pytest.mark.after_training_single_run
def test_is_cross_validation_child_run_single_run() -> None:
    """
    Test that cross validation child runs are identified correctly. A single run should not be identified as a
    cross validation run.
    """
    run = get_most_recent_run()
    # check for offline run
    assert not is_cross_validation_child_run(Run.get_context())
    # check for online runs
    assert not is_cross_validation_child_run(run)


@pytest.mark.after_training_ensemble_run
def test_is_cross_validation_child_run_ensemble_run() -> None:
    """
    Test that cross validation child runs are identified correctly.
    """
    # check for offline run
    assert not is_cross_validation_child_run(Run.get_context())
    # check for online runs
    run = get_most_recent_run(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
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
    # Spurious test failures on Linux build agents, saying that they can't write the file. Wait a bit.
    time.sleep(0.5)
    file1 = test_output_dirs.root_dir / "env1.yml"
    file1.write_text(env1)
    file2 = test_output_dirs.root_dir / "env2.yml"
    file2.write_text(env2)
    # Spurious test failures on Linux build agents, saying that they can't read the file. Wait a bit.
    time.sleep(0.5)
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
    conda_dep, _ = merge_conda_dependencies(files)
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
    get_run_and_check(get_most_recent_run_id(), True, workspace)


@pytest.mark.after_training_ensemble_run
def test_is_completed_ensemble_run() -> None:
    """
    Test if we can correctly check run status and status of child runs for an ensemble run.
    :return:
    """
    logging_to_stdout()
    workspace = get_default_workspace()
    run_id = get_most_recent_run_id(fallback_run_id_for_local_execution=FALLBACK_ENSEMBLE_RUN)
    get_run_and_check(run_id, True, workspace)


def test_amlignore() -> None:
    """
    Test if the private settings files are excluded from getting into the AML snapshot.
    """
    amlignore = repository_root_directory(".amlignore")
    assert amlignore.is_file()
    ignored = amlignore.read_text()
    private_settings = repository_root_directory(PRIVATE_SETTINGS_FILE)
    if private_settings.is_file():
        assert PRIVATE_SETTINGS_FILE in ignored, f"{PRIVATE_SETTINGS_FILE} is not in .amlignore"
    test_variables = repository_root_directory(PROJECT_SECRETS_FILE)
    if test_variables.is_file():
        assert PROJECT_SECRETS_FILE in ignored, f"{PROJECT_SECRETS_FILE} is not in .amlignore"


def test_create_python_env() -> None:
    """
    Checks if environment variables in the SourceConfig are correctly passed through to the Python environment.
    Environment variables in SourceConfig are only used in the internal InnerEye repo.
    :return:
    """
    foo = "foo"
    bar = "bar"
    entry_script = Path("something.py")
    conda_file = get_environment_yaml_file()
    s = SourceConfig(root_folder=Path(""), entry_script=entry_script, conda_dependencies_files=[conda_file],
                     environment_variables={foo: bar})
    env = get_or_create_python_environment(source_config=s,
                                           azure_config=get_default_azure_config(),
                                           register_environment=False)
    assert foo in env.environment_variables
    assert env.environment_variables[foo] == bar

    # Check that some of the basic packages that we expect to always exist are picked up correctly in the Conda env
    def remove_version_number(items: Iterator[str]) -> Set[str]:
        return set(c.split("=")[0] for c in items)

    assert "pytorch" in remove_version_number(env.python.conda_dependencies.conda_packages)
    assert "pytorch-lightning" in remove_version_number(env.python.conda_dependencies.pip_packages)

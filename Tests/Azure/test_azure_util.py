#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest
from azureml.core import Run

from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX, fetch_child_runs, fetch_run, \
    get_cross_validation_split_index, is_cross_validation_child_run, merge_conda_dependencies, \
    to_azure_friendly_container_path
from InnerEye.Common.output_directories import TestOutputDirectories
from Tests.Common.test_util import DEFAULT_ENSEMBLE_RUN_RECOVERY_ID, DEFAULT_ENSEMBLE_RUN_RECOVERY_ID_NUMERIC, \
    DEFAULT_RUN_RECOVERY_ID, DEFAULT_RUN_RECOVERY_ID_NUMERIC
from Tests.ML.util import get_default_workspace


def test_os_path_to_azure_friendly_container_path() -> None:
    """
    Check paths correctly converted to single forward slash formats expected by Azure.
    """
    assert "a" == to_azure_friendly_container_path(Path("a"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a/b/c"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a//b//c"))
    assert "a/b/c" == to_azure_friendly_container_path(Path("a\\b/c/"))


@pytest.mark.parametrize("is_ensemble", [True, False])
def test_get_cross_validation_split_index(is_ensemble: bool) -> None:
    """
    Test that retrieved cross validation split index is as expected, for single runs and ensembles.
    """
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=DEFAULT_ENSEMBLE_RUN_RECOVERY_ID if is_ensemble else DEFAULT_RUN_RECOVERY_ID
    )
    # check for offline run
    assert get_cross_validation_split_index(Run.get_context()) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    # check for online runs
    assert get_cross_validation_split_index(run) == DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    if is_ensemble:
        assert all([get_cross_validation_split_index(x) > DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
                    for x in fetch_child_runs(run)])


@pytest.mark.parametrize("is_ensemble", [True, False])
@pytest.mark.parametrize("is_numeric", [True, False])
def test_is_cross_validation_child_run(is_ensemble: bool, is_numeric: bool) -> None:
    """
    Test that cross validation child runs are identified correctly.
    """
    if is_ensemble:
        rid = DEFAULT_ENSEMBLE_RUN_RECOVERY_ID_NUMERIC if is_numeric else DEFAULT_ENSEMBLE_RUN_RECOVERY_ID
    else:
        rid = DEFAULT_RUN_RECOVERY_ID_NUMERIC if is_numeric else DEFAULT_RUN_RECOVERY_ID
    run = fetch_run(
        workspace=get_default_workspace(),
        run_recovery_id=rid
    )
    # check for offline run
    assert not is_cross_validation_child_run(Run.get_context())
    # check for online runs
    assert not is_cross_validation_child_run(run)
    if is_ensemble:
        assert all([is_cross_validation_child_run(x) for x in fetch_child_runs(run)])


def test_merge_conda(test_output_dirs: TestOutputDirectories) -> None:
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
    file1 = Path(test_output_dirs.root_dir) / "env1.yml"
    file1.write_text(env1)
    file2 = Path(test_output_dirs.root_dir) / "env2.yml"
    file2.write_text(env2)
    conda_dep = merge_conda_dependencies([file1, file2])
    # We expect to see the union of channels.
    assert list(conda_dep.conda_channels) == ["defaults", "pytorch"]
    # Conda package version conflicts are not resolved, but both versions are retained.
    assert list(conda_dep.conda_packages) == ["conda1=1.0", "conda2=2.0", "conda_both=3.0", "conda1=1.1"]
    # For pip packages, the version in the second argument takes precedence.
    assert list(conda_dep.pip_packages) == ["foo==1.0", "azureml-sdk==1.6.0", "bar==2.0"]

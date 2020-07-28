#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pytest

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.build_config import BUILDINFORMATION_JSON, ExperimentResultLocation, \
    build_information_to_dot_net_json, build_information_to_dot_net_json_file
from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.scalar_config import ScalarModelBase


def test_build_config(test_output_dirs: TestOutputDirectories) -> None:
    """
    Test that json with build information is created correctly.
    """
    config = AzureConfig(
        build_number=42,
        build_user="user",
        build_branch="branch",
        build_source_id="00deadbeef",
        build_source_author="author",
        tag="tag",
        model="model")
    result_location = ExperimentResultLocation(azure_job_name="job")
    net_json = build_information_to_dot_net_json(config, result_location)
    expected = '{"BuildNumber": 42, "BuildRequestedFor": "user", "BuildSourceBranchName": "branch", ' \
               '"BuildSourceVersion": "00deadbeef", "BuildSourceAuthor": "author", "ModelName": "model", ' \
               '"ResultsContainerName": null, "ResultsUri": null, "DatasetFolder": null, "DatasetFolderUri": null, ' \
               '"AzureBatchJobName": "job"}'
    assert expected == net_json
    result_folder = Path(test_output_dirs.root_dir) / "buildinfo"
    build_information_to_dot_net_json_file(config, result_location, folder=result_folder)
    result_file = result_folder / BUILDINFORMATION_JSON
    assert result_file.exists()
    assert result_file.read_text() == expected


def test_fields_are_set() -> None:
    """
    Tests that expected fields are set when creating config classes.
    """
    expected = [("hello", None), ("world", None)]
    config = SegmentationModelBase(
        should_validate=False,
        ground_truth_ids=[x[0] for x in expected],
        largest_connected_component_foreground_classes=expected
    )
    assert hasattr(config, CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY)
    assert config.largest_connected_component_foreground_classes == expected


def test_config_non_overridable_params() -> None:
    """
    Check error raised if attempt to override non overridable configs
    """
    non_overridable_params = {k: v.default for k, v in ModelConfigBase.params().items()
                              if k not in ModelConfigBase.get_overridable_parameters()}
    with pytest.raises(ValueError) as ex:
        ModelConfigBase(
            should_validate=False,
            **non_overridable_params
        )
        assert "The following parameters cannot be overriden" in ex.value.args[0]


@pytest.mark.gpu
def test_dataset_reader_workers() -> None:
    """
    Test to make sure the number of dataset reader workers are set correctly
    """
    config = ScalarModelBase(
        should_validate=False,
        num_dataset_reader_workers=-1
    )
    if config.is_offline_run:
        assert config.num_dataset_reader_workers == -1
    else:
        assert config.num_dataset_reader_workers == 0



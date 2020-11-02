#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any

import pytest
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, \
    choice, \
    uniform

from InnerEye.Azure.azure_config import AzureConfig, SourceConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, \
    CROSS_VALIDATION_SUB_FOLD_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.build_config import BUILDINFORMATION_JSON, ExperimentResultLocation, \
    build_information_to_dot_net_json, build_information_to_dot_net_json_file
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.common import TrackedMetrics
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.scalar_config import ScalarModelBase

HYPERDRIVE_TOTAL_RUNS = 64


class HyperDriveTestModelSegmentation(SegmentationModelBase):
    def __init__(self, **params: Any):
        super().__init__(should_validate=False, **params)

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return _create_dummy_hyperdrive_param_search_config(estimator)


class HyperDriveTestModelScalar(ScalarModelBase):
    def __init__(self, **params: Any):
        super().__init__(should_validate=False, **params)

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return _create_dummy_hyperdrive_param_search_config(estimator)


def test_build_config(test_output_dirs: OutputFolderForTests) -> None:
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
    result_folder = test_output_dirs.root_dir / "buildinfo"
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


def test_config_with_typo() -> None:
    with pytest.raises(ValueError) as ex:
        ModelConfigBase(num_epochsi=100)
    assert "The following parameters do not exist: ['num_epochsi']" in ex.value.args[0]


@pytest.mark.cpu_and_gpu
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


@pytest.mark.parametrize("number_of_cross_validation_splits_per_fold", [0, 2])
def test_get_total_number_of_cross_validation_runs(number_of_cross_validation_splits_per_fold: int) -> None:
    config = ScalarModelBase(should_validate=False)
    config.number_of_cross_validation_splits = 2
    config.number_of_cross_validation_splits_per_fold = number_of_cross_validation_splits_per_fold
    assert config.perform_cross_validation

    if number_of_cross_validation_splits_per_fold > 0:
        assert config.perform_sub_fold_cross_validation
        assert config.get_total_number_of_cross_validation_runs() \
               == config.number_of_cross_validation_splits * number_of_cross_validation_splits_per_fold
    else:
        assert not config.perform_sub_fold_cross_validation
        assert config.get_total_number_of_cross_validation_runs() == config.number_of_cross_validation_splits


@pytest.mark.parametrize("number_of_cross_validation_splits", [0, 2])
@pytest.mark.parametrize("number_of_cross_validation_splits_per_fold", [0, 2])
def test_get_hyperdrive_config(number_of_cross_validation_splits: int,
                               number_of_cross_validation_splits_per_fold: int,
                               test_output_dirs: OutputFolderForTests) -> None:
    """
    Test to make sure the number of dataset reader workers are set correctly
    """
    if number_of_cross_validation_splits_per_fold > 0:
        config = HyperDriveTestModelScalar()
        config.number_of_cross_validation_splits_per_fold = number_of_cross_validation_splits_per_fold

    else:
        config = HyperDriveTestModelSegmentation()

    config.number_of_cross_validation_splits = number_of_cross_validation_splits
    # create HyperDrive config with dummy estimator for testing
    source_config = SourceConfig(root_folder=test_output_dirs.root_dir,
                                 entry_script=Path("something.py"), conda_dependencies_files=[])
    estimator = Estimator(
        source_directory=str(source_config.root_folder),
        entry_script=str(source_config.entry_script),
        compute_target="Local"
    )

    hd_config = config.get_hyperdrive_config(estimator=estimator)

    assert hd_config.estimator.source_directory == str(source_config.root_folder)
    assert hd_config.estimator.run_config.script == str(source_config.entry_script)
    assert hd_config.estimator._script_params == source_config.script_params

    if number_of_cross_validation_splits > 0 and number_of_cross_validation_splits_per_fold > 0:
        assert hd_config._max_total_runs == number_of_cross_validation_splits * \
               number_of_cross_validation_splits_per_fold
    elif number_of_cross_validation_splits > 0:
        assert hd_config._max_total_runs == number_of_cross_validation_splits
    else:
        assert hd_config._max_total_runs == HYPERDRIVE_TOTAL_RUNS

    if config.perform_cross_validation:
        # check sampler is as expected
        sampler = config.get_cross_validation_hyperdrive_sampler()

        expected_sampler_dict = {
            CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: choice(list(range(number_of_cross_validation_splits)))
        }

        if number_of_cross_validation_splits_per_fold > 0:
            expected_sampler_dict[CROSS_VALIDATION_SUB_FOLD_SPLIT_INDEX_TAG_KEY] = choice(list(range(
                number_of_cross_validation_splits_per_fold)))

        assert sampler._parameter_space == expected_sampler_dict
    else:
        assert vars(config.get_hyperdrive_config(estimator)) \
               == vars(_create_dummy_hyperdrive_param_search_config(estimator))


def _create_dummy_hyperdrive_param_search_config(estimator: Estimator) -> HyperDriveConfig:
    return HyperDriveConfig(
        estimator=estimator,
        hyperparameter_sampling=RandomParameterSampling({
            'l_rate': uniform(0.0005, 0.01)
        }),
        primary_metric_name=TrackedMetrics.Val_Loss.value,
        primary_metric_goal=PrimaryMetricGoal.MINIMIZE,
        max_total_runs=HYPERDRIVE_TOTAL_RUNS
    )

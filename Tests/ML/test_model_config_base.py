#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest
from typing import Any
from pathlib import Path

from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import HyperDriveConfig, PrimaryMetricGoal, RandomParameterSampling, \
    choice, uniform

from InnerEye.Azure.azure_config import SourceConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.lightning_models import TrackedMetrics
from InnerEye.ML.model_config_base import ModelConfigBase

HYPERDRIVE_TOTAL_RUNS = 64


class HyperDriveTestModel(ModelConfigBase):
    def __init__(self, **params: Any):
        super().__init__(should_validate=False, **params)

    def get_parameter_search_hyperdrive_config(self, estimator: Estimator) -> HyperDriveConfig:
        return _create_dummy_hyperdrive_param_search_config(estimator)


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


@pytest.mark.parametrize("number_of_cross_validation_splits", [0, 2])
def test_get_hyperdrive_config(number_of_cross_validation_splits: int,
                               test_output_dirs: OutputFolderForTests) -> None:
    """
    Test to make sure the number of dataset reader workers are set correctly
    """
    config = HyperDriveTestModel()

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

    if number_of_cross_validation_splits > 0:
        assert hd_config._max_total_runs == number_of_cross_validation_splits
    else:
        assert hd_config._max_total_runs == HYPERDRIVE_TOTAL_RUNS

    if config.perform_cross_validation:
        # check sampler is as expected
        sampler = config.get_cross_validation_hyperdrive_sampler()

        expected_sampler_dict = {
            CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: choice(list(range(number_of_cross_validation_splits)))
        }

        assert sampler._parameter_space == expected_sampler_dict
    else:
        assert vars(config.get_hyperdrive_config(estimator)) \
               == vars(_create_dummy_hyperdrive_param_search_config(estimator))


def test_get_total_number_of_cross_validation_runs() -> None:
    config = ModelConfigBase(should_validate=False)
    config.number_of_cross_validation_splits = 2
    assert config.perform_cross_validation
    assert config.get_total_number_of_cross_validation_runs() == config.number_of_cross_validation_splits


def test_config_with_typo() -> None:
    with pytest.raises(ValueError) as ex:
        ModelConfigBase(num_epochsi=100)
    assert "The following parameters do not exist: ['num_epochsi']" in ex.value.args[0]


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

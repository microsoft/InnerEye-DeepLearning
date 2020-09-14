#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from unittest import mock

from InnerEye.Common import fixed_paths
from InnerEye.ML.normalize_and_visualize_dataset import get_configs
from Tests.ML.configs.DummyModel import DummyModel


def test_visualize_commandline1() -> None:
    """
    Testing for a bug in commandline processing: The model configuration was always overwritten with all the default
    values of each field in Config, rather than only the overrides specified on the commandline.
    :return:
    """
    default_config = DummyModel()
    old_photonorm = default_config.norm_method
    old_random_seed = default_config.get_effective_random_seed()
    new_dataset = "new_dataset"
    assert default_config.azure_dataset_id != new_dataset
    with mock.patch("sys.argv", ["", f"--azure_dataset_id={new_dataset}"]):
        updated_config, runner_config, _ = get_configs(default_config, yaml_file_path=fixed_paths.TRAIN_YAML_FILE)
    assert updated_config.azure_dataset_id == new_dataset
    # These two values were not specified on the commandline, and should be at their original values.
    assert updated_config.norm_method == old_photonorm
    assert updated_config.get_effective_random_seed() == old_random_seed
    # Credentials and variables should have been picked up from yaml files
    assert len(runner_config.datasets_container) > 0
    assert len(runner_config.storage_account) > 0

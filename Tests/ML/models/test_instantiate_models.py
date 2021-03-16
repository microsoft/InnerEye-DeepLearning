#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import List
from unittest import mock

import pytest
from azureml.core import Run

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout, namespace_to_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.model_training import generate_and_print_model_summary, model_train
from InnerEye.ML.runner import Runner
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.model_util import create_model_with_temperature_scaling
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.configs.lightning_test_containers import DummyContainerWithInvalidTrainerArguments, \
    DummyContainerWithModel
from Tests.ML.util import get_default_checkpoint_handler, get_model_loader


def find_models() -> List[str]:
    """
    Lists all Python files in the configs folder. Each of them is assumed to contain one model config.
    :return: list of models
    """
    path = namespace_to_path(ModelConfigLoader.get_default_search_module())
    folders = [path / "segmentation", path / "classification", path / "regression"]
    names = [str(f.stem) for folder in folders for f in folder.glob("*.py") if folder.exists()]
    return [name for name in names if not name.endswith("Base") and not name.startswith("__")]


def test_any_models_found() -> None:
    """
    Test that the basic setup for finding all model configs works: At least one of
    the models are are in the main branch must be found.
    """
    model_names = find_models()
    assert len(model_names) > 0
    assert "Lung" in model_names
    # Test that all configs in the classification folder are picked up as well
    assert "DummyClassification" in model_names


@pytest.mark.parametrize("model_name", find_models())
@pytest.mark.gpu
def test_load_all_configs(model_name: str) -> None:
    """
    Loads all model configurations that are present in the ML/src/configs folder,
    and carries out basic validations of the configuration.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    config = ModelConfigLoader().create_model_config_from_name(model_name)
    assert config.model_name == model_name, "Mismatch between definition .py file and model name"
    if config.is_segmentation_model:
        # Reduce the feature channels to a minimum, to make tests run fast on CPU.
        minimal_feature_channels = 1
        config.feature_channels = [minimal_feature_channels] * len(config.feature_channels)
        print("Model architecture after restricting to 2 feature channels only:")
        model = create_model_with_temperature_scaling(config)
        generate_and_print_model_summary(config, model)  # type: ignore
    else:
        # For classification models, we can't always print a model summary: The model could require arbitrary
        # numbers of input tensors, and we'd only know once we load the training data.
        # Hence, only try to create the model, but don't attempt to print the summary.
        create_model_with_temperature_scaling(config)


def test_cross_validation_config() -> None:
    CrossValidationDummyModel(0, -1)
    CrossValidationDummyModel(10, 1)
    CrossValidationDummyModel(10, -1)

    with pytest.raises(ValueError):
        CrossValidationDummyModel(10, 11)
    with pytest.raises(ValueError):
        CrossValidationDummyModel(10, 10)


class CrossValidationDummyModel(DummyModel):
    def __init__(self, number_of_cross_validation_splits: int, cross_validation_split_index: int):
        self.number_of_cross_validation_splits = number_of_cross_validation_splits
        self.cross_validation_split_index = cross_validation_split_index
        super().__init__()


def test_model_config_loader() -> None:
    logging_to_stdout(log_level=logging.DEBUG)
    default_loader = get_model_loader()
    assert default_loader.create_model_config_from_name("BasicModel2Epochs") is not None
    with pytest.raises(ValueError):
        default_loader.create_model_config_from_name("DummyModel")
    loader_including_tests = get_model_loader(namespace="Tests.ML.configs")
    assert loader_including_tests.create_model_config_from_name("BasicModel2Epochs") is not None
    assert loader_including_tests.create_model_config_from_name("DummyModel") is not None


def test_config_loader_as_in_registration() -> None:
    """
    During model registration, the model config namespace is read out from the present model. Ensure that we
    can create a config loader that has that value as an input.
    """
    loader1 = ModelConfigLoader()
    model_name = "BasicModel2Epochs"
    model = loader1.create_model_config_from_name(model_name)
    assert model is not None
    namespace = model.__module__
    loader2 = ModelConfigLoader(model_configs_namespace=namespace)
    assert len(loader2.module_search_specs) == 2
    model2 = loader2.create_model_config_from_name(model_name)
    assert model2 is not None


def test_config_loader_on_lightning_container() -> None:
    """
    Test if the config loader can load an model that is neither classification nor segmentation.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    loader_including_tests = get_model_loader(namespace="Tests.ML.configs")
    model = loader_including_tests.create_model_config_from_name("DummyLightningContainer")
    assert model is not None


@pytest.mark.parametrize("container_name", ["DummyContainerWithAzureDataset",
                                            "DummyContainerWithoutDataset",
                                            "DummyContainerWithLocalDataset",
                                            "DummyContainerWithAzureAndLocalDataset"])
def test_submit_container_to_azureml(container_name: str) -> None:
    """
    Test if we can get the config loader to load a Lightning container model, and get it through the AzureML
    submission process.
    """
    runner = Runner(project_root=fixed_paths.repository_root_directory(),
                    yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
    mock_run = Run.get_context()
    args = ["", f"--model={container_name}", "--azureml=True", "--model_configs_namespace=Tests.ML.configs"]
    with mock.patch("sys.argv", args):
        with mock.patch("InnerEye.Azure.azure_runner.get_dataset_consumption", return_value=None):
            with mock.patch("azureml.core.Experiment.submit", return_value=mock_run):
                loaded_config, actual_run = runner.run()
    assert actual_run == mock_run
    assert isinstance(loaded_config, DeepLearningConfig)
    assert isinstance(runner.lightning_container, LightningContainer)


def test_run_container_in_situ() -> None:
    """
    Test if we can get the config loader to load a Lightning container model, and then train locally.
    """
    runner = Runner(project_root=fixed_paths.repository_root_directory(),
                    yaml_config_file=fixed_paths.SETTINGS_YAML_FILE)
    args = ["", "--model=DummyContainerWithModel", "--model_configs_namespace=Tests.ML.configs"]
    with mock.patch("sys.argv", args):
        loaded_config, actual_run = runner.run()
    assert actual_run is None
    assert isinstance(runner.lightning_container, DummyContainerWithModel)


def test_run_model_with_invalid_trainer_arguments(test_output_dirs: OutputFolderForTests) -> None:
    container = DummyContainerWithInvalidTrainerArguments()
    config = container.create_lightning_module()
    container.lightning_module = config
    checkpoint_handler = get_default_checkpoint_handler(model_config=config,
                                                        project_root=test_output_dirs.root_dir)
    with pytest.raises(Exception) as ex:
        model_train(container.lightning_module, checkpoint_handler=checkpoint_handler, lightning_container=container)
    assert "no_such_argument" in str(ex)

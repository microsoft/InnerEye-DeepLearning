#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from typing import List

import pytest

from InnerEye.Common.common_util import logging_to_stdout, namespace_to_path
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_training import generate_and_print_model_summary
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.utils.model_util import create_model_with_temperature_scaling
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_model_loader


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
    config = ModelConfigLoader[SegmentationModelBase]().create_model_config_from_name(model_name)
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
    loader1 = ModelConfigLoader[SegmentationModelBase]()
    model_name = "BasicModel2Epochs"
    model = loader1.create_model_config_from_name(model_name)
    assert model is not None
    namespace = model.__module__
    loader2 = ModelConfigLoader[SegmentationModelBase](model_configs_namespace=namespace)
    assert len(loader2.module_search_specs) == 2
    model2 = loader2.create_model_config_from_name(model_name)
    assert model2 is not None


def test_create_lightning_container() -> None:
    """
    Test if we can instantiate the container object itself, without any issues with inheritance or metaclasses
    """
    from InnerEye.ML.lightning_container import LightningContainer
    LightningContainer()


def test_config_loader_on_lightning_container() -> None:
    """
    Test if the config loader can load an model that is neither classification nor segmentation.
    """
    logging_to_stdout(log_level=logging.DEBUG)
    loader_including_tests = get_model_loader(namespace="Tests.ML.configs")
    model = loader_including_tests.create_model_config_from_name("DummyLightningContainer")
    assert model is not None

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import numpy as np
import pytest
import torch

from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.pipelines.scalar_inference import ScalarEnsemblePipeline, ScalarInferencePipeline, \
    ScalarInferencePipelineBase
from InnerEye.ML.scalar_config import EnsembleAggregationType
from InnerEye.ML.utils.model_util import ModelAndInfo
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.fixed_paths_for_tests import full_ml_test_data_path


def test_create_from_checkpoint_non_ensemble() -> None:
    config = ClassificationModelForTesting()

    # when checkpoint does not exist, return None
    checkpoint_folder = "classification_data_generated_random/checkpoints/non_exist.pth.tar"
    path_to_checkpoint = full_ml_test_data_path(checkpoint_folder)
    inference_pipeline = ScalarInferencePipeline.create_from_checkpoint(path_to_checkpoint, config)
    assert inference_pipeline is None

    checkpoint_folder = "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar"
    path_to_checkpoint = full_ml_test_data_path(checkpoint_folder)
    inference_pipeline = ScalarInferencePipeline.create_from_checkpoint(path_to_checkpoint, config)
    assert isinstance(inference_pipeline, ScalarInferencePipeline)
    assert inference_pipeline.epoch == 1


def test_create_from_checkpoint_ensemble() -> None:
    config = ClassificationModelForTesting()

    checkpoint_folder_non_exist = "classification_data_generated_random/checkpoints/non_exist.pth.tar"
    path_to_checkpoint_non_exist = full_ml_test_data_path(checkpoint_folder_non_exist)
    checkpoint_folder_exist = "classification_data_generated_random/checkpoints/1_checkpoint.pth.tar"
    path_to_checkpoint_exist = full_ml_test_data_path(checkpoint_folder_exist)

    # when all checkpoints do not exist, raise error
    with pytest.raises(ValueError):
        paths_to_checkpoint = [path_to_checkpoint_non_exist] * 5
        ScalarEnsemblePipeline.create_from_checkpoint(paths_to_checkpoint, config)

    # when a few checkpoints exist, ensemble with those
    paths_to_checkpoint = [path_to_checkpoint_non_exist] * 3 + [path_to_checkpoint_exist] * 2
    inference_pipeline = ScalarEnsemblePipeline.create_from_checkpoint(paths_to_checkpoint, config)
    assert isinstance(inference_pipeline, ScalarEnsemblePipeline)
    assert len(inference_pipeline.pipelines) == 2

    # when all checkpoints exist
    paths_to_checkpoint = [path_to_checkpoint_exist] * 5
    inference_pipeline = ScalarEnsemblePipeline.create_from_checkpoint(paths_to_checkpoint, config)
    assert isinstance(inference_pipeline, ScalarEnsemblePipeline)
    assert len(inference_pipeline.pipelines) == 5


def test_create_result_dataclass() -> None:
    # invalid instances: these try to instantiate with inconsistent length lists/tensors
    with pytest.raises(ValueError):
        # one sample, but labels has length 2
        ScalarInferencePipelineBase.Result(['1'], torch.zeros((2, 1)), torch.zeros((1, 1, 2, 2, 2)))
    with pytest.raises(ValueError):
        # one sample, but model output has batch size 2
        ScalarInferencePipelineBase.Result(['1'], torch.zeros((1, 1)), torch.zeros((2, 1, 2, 2, 2)))

    # these are valid instances
    ScalarInferencePipelineBase.Result(['1'], torch.zeros((1, 1)), torch.zeros((1, 1, 2, 2, 2)))
    ScalarInferencePipelineBase.Result(['1', '2'], torch.zeros((2, 1)), torch.zeros((2, 1, 2, 2, 2)))


# Mock model, always predicts the same scalar value
class ConstantScalarModel(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self, expected_image_size_zyx: TupleInt3, scalar_to_return: float) -> None:
        super().__init__()
        self.expected_image_size_zyx = expected_image_size_zyx
        self._layers = torch.nn.ModuleList([])
        self.return_value = scalar_to_return

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        """
        Transforms a classification item into images
        :param item: ClassificationItem
        :return: Tensor
        """
        return [item.images]

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        if x.shape[-3:] != self.expected_image_size_zyx:
            raise ValueError(f"Expected a tensor with trailing size {self.expected_image_size_zyx}, but got "
                             f"{x.shape}")

        return torch.full((x.shape[0], 1), self.return_value)


class ConstantScalarConfig(ClassificationModelForTesting):
    def __init__(self, scalar_to_return: float):
        super().__init__()
        self.scalar_to_return = scalar_to_return

    def create_model(self) -> Any:
        return ConstantScalarModel(expected_image_size_zyx=self.expected_image_size_zyx,
                                   scalar_to_return=self.scalar_to_return)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize("empty_labels", [True, False])
def test_predict_non_ensemble(batch_size: int, empty_labels: bool) -> None:
    config = ConstantScalarConfig(1.)
    model_and_info = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
                                  is_mean_teacher=False, checkpoint_path=None)
    model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded

    model = model_and_info.model

    pipeline = ScalarInferencePipeline(model, config, 0, 0)
    actual_labels = torch.zeros((batch_size, 1)) * np.nan if empty_labels else torch.zeros((batch_size, 1))
    data = {"metadata": [GeneralSampleMetadata(id='2')] * batch_size,
            "label": actual_labels,
            "images": torch.zeros(((batch_size, 1) + config.expected_image_size_zyx)),
            "numerical_non_image_features": torch.tensor([]),
            "categorical_non_image_features": torch.tensor([]),
            "segmentations": torch.tensor([])}

    results = pipeline.predict(data)
    ids, labels, predicted = results.subject_ids, results.labels, results.model_outputs
    assert ids == ['2'] * batch_size
    assert torch.allclose(labels, actual_labels, equal_nan=True)
    # The model always returns 1, so predicted should be sigmoid(1)
    assert torch.allclose(predicted, torch.full((batch_size, 1), 0.731058578))


@pytest.mark.parametrize('batch_size', [1, 3])
def test_predict_ensemble(batch_size: int) -> None:
    config_returns_0 = ConstantScalarConfig(0.)
    model_and_info_returns_0 = ModelAndInfo(config=config_returns_0, model_execution_mode=ModelExecutionMode.TEST,
                                            is_mean_teacher=False, checkpoint_path=None)
    model_loaded = model_and_info_returns_0.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded
    model_returns_0 = model_and_info_returns_0.model

    config_returns_1 = ConstantScalarConfig(1.)
    model_and_info_returns_1 = ModelAndInfo(config=config_returns_1, model_execution_mode=ModelExecutionMode.TEST,
                                            is_mean_teacher=False, checkpoint_path=None)
    model_loaded = model_and_info_returns_1.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded
    model_returns_1 = model_and_info_returns_1.model

    pipeline_0 = ScalarInferencePipeline(model_returns_0, config_returns_0, 0, 0)
    pipeline_1 = ScalarInferencePipeline(model_returns_0, config_returns_0, 0, 1)
    pipeline_2 = ScalarInferencePipeline(model_returns_0, config_returns_0, 0, 2)
    pipeline_3 = ScalarInferencePipeline(model_returns_1, config_returns_1, 0, 3)
    pipeline_4 = ScalarInferencePipeline(model_returns_1, config_returns_1, 0, 4)
    ensemble_pipeline = ScalarEnsemblePipeline([pipeline_0, pipeline_1, pipeline_2, pipeline_3, pipeline_4],
                                               config_returns_0, EnsembleAggregationType.Average)
    data = {"metadata": [GeneralSampleMetadata(id='2')] * batch_size,
            "label": torch.zeros((batch_size, 1)),
            "images": torch.zeros(((batch_size, 1) + config_returns_0.expected_image_size_zyx)),
            "numerical_non_image_features": torch.tensor([]),
            "categorical_non_image_features": torch.tensor([]),
            "segmentations": torch.tensor([])}

    results = ensemble_pipeline.predict(data)
    ids, labels, predicted = results.subject_ids, results.labels, results.model_outputs
    assert ids == ['2'] * batch_size
    assert torch.equal(labels, torch.zeros((batch_size, 1)))
    # 3 models return 0, 2 return 1, so predicted should be ((sigmoid(0)*3)+(sigmoid(1)*2))/5
    assert torch.allclose(predicted, torch.full((batch_size, 1), 0.592423431))


input_tensor = torch.from_numpy(np.array([[[0.5], [1.0], [0.4]],
                                          [[0.5], [1.0], [0.4]],
                                          [[0.6], [1.0], [0.4]],
                                          [[0.6], [1.0], [0.5]],
                                          [[0.5], [1.0], [0.5]]]))

result_average = torch.from_numpy(np.array([[0.54], [1.0], [0.44]]))


@pytest.mark.parametrize("aggregation_type, expected_result", [(EnsembleAggregationType.Average, result_average)])
def test_aggregation(aggregation_type: EnsembleAggregationType, expected_result: torch.Tensor) -> None:
    output = ScalarEnsemblePipeline.aggregate_model_outputs(input_tensor, aggregation_type)
    assert output.shape == input_tensor[0].shape
    assert torch.allclose(output, expected_result)

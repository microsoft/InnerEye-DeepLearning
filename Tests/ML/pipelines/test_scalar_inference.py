#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import numpy as np
import pytest
import torch

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.dataset.sample import GeneralSampleMetadata
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.lightning_helpers import create_lightning_model
from InnerEye.ML.lightning_models import ScalarLightning
from InnerEye.ML.models.architectures.base_model import DeviceAwareModule
from InnerEye.ML.pipelines.scalar_inference import ScalarEnsemblePipeline, ScalarInferencePipeline, \
    ScalarInferencePipelineBase
from InnerEye.ML.scalar_config import EnsembleAggregationType
from Tests.ML.configs.ClassificationModelForTesting import ClassificationModelForTesting
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint


def test_create_from_checkpoint_non_ensemble(test_output_dirs: OutputFolderForTests) -> None:
    config = ClassificationModelForTesting()

    # when checkpoint does not exist, return None
    path_to_checkpoint = test_output_dirs.root_dir / "foo.ckpt"
    inference_pipeline = ScalarInferencePipeline.create_from_checkpoint(path_to_checkpoint, config)
    assert inference_pipeline is None

    create_model_and_store_checkpoint(config, path_to_checkpoint)
    inference_pipeline = ScalarInferencePipeline.create_from_checkpoint(path_to_checkpoint, config)
    assert isinstance(inference_pipeline, ScalarInferencePipeline)


def test_create_from_checkpoint_ensemble(test_output_dirs: OutputFolderForTests) -> None:
    config = ClassificationModelForTesting()

    path_to_checkpoint_non_exist = test_output_dirs.root_dir / "does_not_exist.ckpt"
    path_to_checkpoint_exist = test_output_dirs.root_dir / "foo.ckpt"
    create_model_and_store_checkpoint(config, path_to_checkpoint_exist)

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
    model = create_lightning_model(config, set_optimizer_and_scheduler=False)
    assert isinstance(model, ScalarLightning)

    pipeline = ScalarInferencePipeline(model, config, 0)
    actual_labels = torch.zeros((batch_size, 1)) * np.nan if empty_labels else torch.zeros((batch_size, 1))
    data = {"metadata": [GeneralSampleMetadata(id='2')] * batch_size,
            "label": actual_labels,
            "images": torch.zeros(((batch_size, 1) + config.expected_image_size_zyx)),
            "numerical_non_image_features": torch.tensor([]),
            "categorical_non_image_features": torch.tensor([]),
            "segmentations": torch.tensor([])}

    results = pipeline.predict(data)
    ids, labels, predicted = results.subject_ids, results.labels, results.posteriors
    assert ids == ['2'] * batch_size
    assert torch.allclose(labels, actual_labels, equal_nan=True)
    # The model always returns 1, so predicted should be sigmoid(1)
    assert torch.allclose(predicted, torch.full((batch_size, 1), 0.731058578))


@pytest.mark.parametrize('batch_size', [1, 3])
def test_predict_ensemble(batch_size: int) -> None:
    config_returns_0 = ConstantScalarConfig(0.)
    model_returns_0 = create_lightning_model(config_returns_0, set_optimizer_and_scheduler=False)
    assert isinstance(model_returns_0, ScalarLightning)

    config_returns_1 = ConstantScalarConfig(1.)
    model_returns_1 = create_lightning_model(config_returns_1, set_optimizer_and_scheduler=False)
    assert isinstance(model_returns_1, ScalarLightning)

    pipeline_0 = ScalarInferencePipeline(model_returns_0, config_returns_0, 0)
    pipeline_1 = ScalarInferencePipeline(model_returns_0, config_returns_0, 1)
    pipeline_2 = ScalarInferencePipeline(model_returns_0, config_returns_0, 2)
    pipeline_3 = ScalarInferencePipeline(model_returns_1, config_returns_1, 3)
    pipeline_4 = ScalarInferencePipeline(model_returns_1, config_returns_1, 4)
    ensemble_pipeline = ScalarEnsemblePipeline([pipeline_0, pipeline_1, pipeline_2, pipeline_3, pipeline_4],
                                               config_returns_0, EnsembleAggregationType.Average)
    data = {"metadata": [GeneralSampleMetadata(id='2')] * batch_size,
            "label": torch.zeros((batch_size, 1)),
            "images": torch.zeros(((batch_size, 1) + config_returns_0.expected_image_size_zyx)),
            "numerical_non_image_features": torch.tensor([]),
            "categorical_non_image_features": torch.tensor([]),
            "segmentations": torch.tensor([])}

    results = ensemble_pipeline.predict(data)
    ids, labels, predicted = results.subject_ids, results.labels, results.posteriors
    assert ids == ['2'] * batch_size
    assert torch.equal(labels, torch.zeros((batch_size, 1)))
    # 3 models return 0, 2 return 1, so predicted should be ((sigmoid(0)*3)+(sigmoid(1)*2))/5
    assert torch.allclose(predicted, torch.full((batch_size, 1), 0.592423431))


def test_aggregation() -> None:
    aggregation_type = EnsembleAggregationType.Average
    pipeline = ScalarEnsemblePipeline(ensemble_aggregation_type=aggregation_type,
                                      pipelines=None,  # type: ignore
                                      model_config=None)  # type: ignore
    input_tensor = torch.tensor([[[0.5], [1.0], [0.4]],
                                 [[0.5], [1.0], [0.4]],
                                 [[0.6], [1.0], [0.4]],
                                 [[0.6], [1.0], [0.5]],
                                 [[0.5], [1.0], [0.5]]])
    result_average = input_tensor.mean(dim=0)
    output = pipeline.aggregate_model_outputs(input_tensor)
    assert output.shape == input_tensor[0].shape
    assert torch.allclose(output, result_average)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Optional, Union

import numpy as np
import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.configs.classification.DummyClassification import DummyClassification
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.model_training import model_train
from InnerEye.ML.models.architectures.base_model import BaseModel, CropSizeConstraints
from InnerEye.ML.models.parallel.data_parallel import DataParallelModel
from InnerEye.ML.pipelines.forward_pass import SegmentationForwardPass
from InnerEye.ML.utils import ml_util, model_util
from InnerEye.ML.utils.io_util import ImageDataType
from InnerEye.ML.utils.model_util import ModelAndInfo, create_model_with_temperature_scaling
from Tests.ML.util import machine_has_gpu, no_gpu_available


class SimpleModel(BaseModel):
    def __init__(self, input_channels: int, channels: list, n_classes: int, kernel_size: int,
                 insert_value_in_output: Optional[float] = None,
                 crop_size_constraints: CropSizeConstraints = None):
        super().__init__(input_channels=input_channels, name="SimpleModel", crop_size_constraints=crop_size_constraints)
        self.channels = channels
        self.n_classes = n_classes
        self.kernel_size = kernel_size
        self.insert_value_in_output = insert_value_in_output
        self._model = torch.nn.Sequential(
            torch.nn.Conv3d(input_channels, channels[0], kernel_size=self.kernel_size),
            torch.nn.ConvTranspose3d(channels[0], n_classes, kernel_size=self.kernel_size)
        )

    def forward(self, x: Any) -> Any:  # type: ignore
        x = self._model(x)
        if self.insert_value_in_output:
            x[..., 0] = self.insert_value_in_output
        return x

    def get_all_child_layers(self) -> List[torch.nn.Module]:
        return list(self._model.children())


@pytest.mark.parametrize("value_to_insert", [1.0, np.NaN, np.Inf])
@pytest.mark.parametrize("in_training_mode", [True, False])
def test_anomaly_detection(value_to_insert: float, in_training_mode: bool) -> None:
    """
    Test anomaly detection for the segmentation forward pass.
    :param value_to_insert: The value to insert in the image image (nan, inf, or a valid float)
    :param in_training_mode: If true, run the segmentation forward pass in training mode, otherwise use the
    settings for running on the validation set.
    :return:
    """
    image_size = [1, 1, 4, 4, 4]
    labels_size = [1, 2, 4, 4, 4]
    mask_size = [1, 4, 4, 4]
    crop_size = (4, 4, 4)
    inference_stride_size = (2, 2, 2)
    ground_truth_ids = ["Lung"]

    # image to run inference on
    image = torch.from_numpy(np.random.uniform(size=image_size).astype(ImageDataType.IMAGE.value))
    # labels for criterion
    labels = torch.from_numpy(np.random.uniform(size=labels_size).astype(ImageDataType.SEGMENTATION.value))
    # create a random mask if required
    mask = torch.from_numpy((np.round(np.random.uniform(size=mask_size)).astype(dtype=ImageDataType.MASK.value)))

    config = SegmentationModelBase(
        crop_size=crop_size,
        inference_stride_size=inference_stride_size,
        image_channels=["ct"],
        ground_truth_ids=ground_truth_ids,
        should_validate=False,
        detect_anomaly=True
    )

    # instantiate the model
    model = SimpleModel(1, [1], 2, 2)
    config.adjust_after_mixed_precision_and_parallel(model)
    config.use_gpu = False

    # Create the optimizer_type and loss criterion
    optimizer = model_util.create_optimizer(config, model)
    criterion = lambda x, y: torch.tensor(value_to_insert, requires_grad=True)
    pipeline = SegmentationForwardPass(model,
                                       config,
                                       batch_size=1,
                                       optimizer=optimizer,
                                       in_training_mode=in_training_mode,
                                       criterion=criterion)
    image[0, 0, 0, 0, 0] = value_to_insert
    if np.isnan(value_to_insert) or np.isinf(value_to_insert):
        with pytest.raises(RuntimeError) as ex:
            pipeline.forward_pass_patches(patches=image, mask=mask, labels=labels)
        assert f"loss computation returned {value_to_insert}" in str(ex)
    else:
        pipeline.forward_pass_patches(patches=image, mask=mask, labels=labels)


@pytest.mark.gpu
@pytest.mark.skipif(no_gpu_available, reason="Testing AMP requires a GPU")
@pytest.mark.parametrize("use_model_parallel", [False, True])
@pytest.mark.parametrize("use_mixed_precision", [False, True])
@pytest.mark.parametrize("execution_mode", [ModelExecutionMode.TRAIN, ModelExecutionMode.TEST])
def test_amp_activated(use_model_parallel: bool,
                       execution_mode: ModelExecutionMode,
                       use_mixed_precision: bool) -> None:
    """
    Tests the amp flag both for True and False states. Verifys that the mixed precision training functions as expected.
    """
    assert machine_has_gpu, "This test must be executed on a GPU machine."
    assert torch.cuda.device_count() > 1, "This test must be executed on a multi-GPU machine"
    # image, labels, and mask to run forward and backward passes
    image = torch.from_numpy(np.random.uniform(size=[1, 1, 4, 4, 4]).astype(ImageDataType.IMAGE.value))
    labels = torch.from_numpy(np.random.uniform(size=[1, 2, 4, 4, 4]).astype(ImageDataType.SEGMENTATION.value))
    mask = torch.from_numpy((np.round(np.random.uniform(size=[1, 4, 4, 4])).astype(dtype=ImageDataType.MASK.value)))

    crop_size = (4, 4, 4)

    model = SimpleModel(1, [1], 2, 2)
    model_config = SegmentationModelBase(crop_size=crop_size,
                                         image_channels=["ct"],
                                         ground_truth_ids=["Lung"],
                                         use_mixed_precision=use_mixed_precision,
                                         use_model_parallel=use_model_parallel,
                                         should_validate=False)
    assert model_config.use_gpu
    # Move the model to the GPU. This is mostly to avoid issues with AMP, which has trouble
    # with first using a GPU model and later using a CPU-based one.
    model = model.cuda()
    optimizer = model_util.create_optimizer(model_config, model)
    model_and_info = ModelAndInfo(model, optimizer)
    try:
        model_and_info_amp = model_util.update_model_for_mixed_precision_and_parallel(model_and_info,
                                                                                      model_config,
                                                                                      execution_mode)
    except NotImplementedError as ex:
        if use_model_parallel:
            # The SimpleModel does not implement model partitioning, and should hence fail at this step.
            assert "Model partitioning is not implemented" in str(ex)
            return
        else:
            raise ValueError(f"Expected this call to succeed, but got: {ex}")

    # Check if the optimizer is updated with AMP mixed precision features. The attribute should be present
    # if and only if mixed precision is switched on.
    optimizer_amp = model_and_info_amp.optimizer
    assert optimizer_amp is not None
    assert hasattr(optimizer_amp, '_amp_stash') == use_mixed_precision
    assert hasattr(optimizer_amp, '_post_amp_backward') == use_mixed_precision

    criterion = lambda x, y: torch.tensor([0.0], requires_grad=True).cuda()
    pipeline = SegmentationForwardPass(model_and_info_amp.model,
                                       model_config,
                                       batch_size=1,
                                       optimizer=optimizer_amp,
                                       criterion=criterion)

    # Verify that forward and backward passes do not throw an exception
    pipeline._forward_pass(patches=image, mask=mask, labels=labels)


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
@pytest.mark.parametrize("use_gpu_override", [False, True])
def test_use_gpu_flag(use_gpu_override: bool) -> None:
    config = DeepLearningConfig(should_validate=False)
    # On a model that does not have a use_gpu_override, the use_gpu flag should return True exactly when a GPU is
    # actually present.
    assert config.use_gpu == machine_has_gpu
    if machine_has_gpu:
        # If a GPU is present, the use_gpu flag should exactly return whatever the override says
        # (we can run in CPU mode even on a GPU)
        config.use_gpu = use_gpu_override
        assert config.use_gpu == use_gpu_override
    else:
        if use_gpu_override:
            # We are on a machine without a GPU, but the override says we should use the GPU: fail.
            with pytest.raises(ValueError) as ex:
                config.use_gpu = use_gpu_override
            assert "use_gpu to True if there is not CUDA capable GPU present" in str(ex)
        else:
            config.use_gpu = use_gpu_override
            assert config.use_gpu == use_gpu_override


@pytest.mark.gpu
def test_mean_teacher_model() -> None:
    """
    Test training and weight updates of the mean teacher model computation.
    """
    def _get_parameters_of_model(model: Union[torch.nn.Module, DataParallelModel]) -> Any:
        """
        Returns the iterator of model parameters
        """
        if isinstance(model, DataParallelModel):
            return model.module.parameters()
        else:
            return model.parameters()

    config = DummyClassification()
    config.num_epochs = 1
    # Set train batch size to be arbitrary big to ensure we have only one training step
    # i.e. one mean teacher update.
    config.train_batch_size = 100
    # Train without mean teacher
    model_train(config)

    # Retrieve the weight after one epoch
    model = create_model_with_temperature_scaling(config)
    print(config.get_path_to_checkpoint(1))
    _ = model_util.load_checkpoint(model, config.get_path_to_checkpoint(1))
    model_weight = next(_get_parameters_of_model(model))

    # Get the starting weight of the mean teacher model
    ml_util.set_random_seed(config.get_effective_random_seed())
    _ = create_model_with_temperature_scaling(config)
    mean_teach_model = create_model_with_temperature_scaling(config)
    initial_weight_mean_teacher_model = next(_get_parameters_of_model(mean_teach_model))

    # Now train with mean teacher and check the update of the weight
    alpha = 0.999
    config.mean_teacher_alpha = alpha
    model_train(config)

    # Retrieve weight of mean teacher model saved in the checkpoint
    mean_teacher_model = create_model_with_temperature_scaling(config)
    _ = model_util.load_checkpoint(mean_teacher_model, config.get_path_to_checkpoint(1, for_mean_teacher_model=True))
    result_weight = next(_get_parameters_of_model(mean_teacher_model))
    # Retrieve the associated student weight
    _ = model_util.load_checkpoint(model, config.get_path_to_checkpoint(1))
    student_model_weight = next(_get_parameters_of_model(model))

    # Assert that the student weight corresponds to the weight of a simple training without mean teacher
    # computation
    assert student_model_weight.allclose(model_weight)

    # Check the update of the parameters
    assert torch.all(alpha * initial_weight_mean_teacher_model + (1 - alpha) * student_model_weight == result_weight)

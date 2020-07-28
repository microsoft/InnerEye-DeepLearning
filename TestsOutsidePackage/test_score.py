#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pytest

from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.util import get_model_loader
from Tests.fixed_paths_for_tests import full_ml_test_data_path
from score import create_inference_pipeline, is_spacing_valid, run_inference

test_image = full_ml_test_data_path("train_and_test_data") / "id1_channel1.nii.gz"
img_nii_path = full_ml_test_data_path("test_img.nii.gz")
checkpoint_full_paths = [full_ml_test_data_path('checkpoints') / "1_checkpoint.pth.tar"]
LOADER = get_model_loader("Tests.ML.configs")


def test_score_check_spacing() -> None:
    config = LOADER.create_model_config_from_name("DummyModel")
    config.dataset_expected_spacing_xyz = (1.0, 1.0, 3.0)
    image_with_header = io_util.load_nifti_image(img_nii_path)
    spacing_xyz = reverse_tuple_float3(image_with_header.header.spacing)
    assert is_spacing_valid(spacing_xyz, config.dataset_expected_spacing_xyz)
    assert is_spacing_valid(spacing_xyz, (1, 1, 3.01))
    assert not is_spacing_valid(spacing_xyz, (1, 1, 3.2))


@pytest.mark.parametrize("is_ensemble", [True, False])
def test_run_scoring(is_ensemble: bool) -> None:
    checkpoints_paths = checkpoint_full_paths * 2 if is_ensemble else checkpoint_full_paths
    dummy_config = DummyModel()
    inference_pipeline, dummy_config = create_inference_pipeline(dummy_config, checkpoints_paths, use_gpu=False)
    image_with_header = io_util.load_nifti_image(test_image)
    result = run_inference([image_with_header, image_with_header], inference_pipeline, dummy_config)
    assert np.all(result == 1)
    assert image_with_header.image.shape == result.shape  # type: ignore

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pytest
from pytorch_lightning import seed_everything

from unittest import mock

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint
from score import create_inference_pipeline, is_spacing_valid, run_inference, score_image, ScorePipelineConfig


test_image = full_ml_test_data_path("train_and_test_data") / "id1_channel1.nii.gz"
img_nii_path = full_ml_test_data_path("test_img.nii.gz")

hnsegmentation_file = "hnsegmentation.nii.gz"
hnsegmentation_path = full_ml_test_data_path(hnsegmentation_file)
sample_dicom_zip_file = full_ml_test_data_path("sample_dicom.zip")


def test_score_check_spacing() -> None:
    config = DummyModel()
    config.dataset_expected_spacing_xyz = (1.0, 1.0, 3.0)
    image_with_header = io_util.load_nifti_image(img_nii_path)
    spacing_xyz = reverse_tuple_float3(image_with_header.header.spacing)
    assert is_spacing_valid(spacing_xyz, config.dataset_expected_spacing_xyz)
    assert is_spacing_valid(spacing_xyz, (1, 1, 3.01))
    assert not is_spacing_valid(spacing_xyz, (1, 1, 3.2))


@pytest.mark.parametrize("is_ensemble", [True, False])
def test_run_scoring(test_output_dirs: OutputFolderForTests, is_ensemble: bool) -> None:
    """
    Run the scoring script on an image file.
    This test lives outside the normal Tests folder because it imports "score.py" from the repository root folder.
    If we switched to InnerEye as a package, we would have to treat this import special.
    The inference run here is on a 1-channel model, whereas test_register_and_score_model works with a 2-channel
    model.
    """
    seed_everything(42)
    checkpoint = test_output_dirs.root_dir / "checkpoint.ckpt"
    image_size = (40, 40, 40)
    test_crop_size = image_size
    dummy_config = DummyModel()
    dummy_config.test_crop_size = test_crop_size
    dummy_config.inference_stride_size = (10, 10, 10)
    dummy_config.inference_batch_size = 10
    create_model_and_store_checkpoint(dummy_config, checkpoint)
    all_paths = [checkpoint] * 2 if is_ensemble else [checkpoint]
    inference_pipeline, dummy_config = create_inference_pipeline(dummy_config, all_paths, use_gpu=False)
    image_with_header = io_util.load_nifti_image(test_image)
    image_with_header.image = image_with_header.image[:image_size[0], :image_size[1], :image_size[2]]
    result = run_inference([image_with_header, image_with_header], inference_pipeline, dummy_config)
    assert image_with_header.image.shape == result.shape  # type: ignore
    print(f"Unique result values: {np.unique(result)}")
    assert np.all(result == 1)


def test_score_image_nifti(test_output_dirs: OutputFolderForTests) -> None:
    model_folder = test_output_dirs.root_dir / "final"

    score_pipeline_config = ScorePipelineConfig(
        data_folder=full_ml_test_data_path(),
        model_folder=str(model_folder),
        image_files=[str(img_nii_path)],
        result_image_name='result_image_name',
        use_gpu=False,
        use_dicom=False)

    with mock.patch('score.init_from_model_inference_json',
                    return_value=(1, 2)):
        with mock.patch('score.run_inference',
                        return_value=True):
            with mock.patch('score.store_as_ubyte_nifti',
                            return_value='result_image_name.nii.gz'):
                segmentation = score_image(score_pipeline_config)
                assert segmentation


def test_score_image_dicom(test_output_dirs: OutputFolderForTests) -> None:
    model_folder = test_output_dirs.root_dir / "final"

    score_pipeline_config = ScorePipelineConfig(
        data_folder=full_ml_test_data_path("HN"),
        model_folder=str(model_folder),
        image_files=[str(sample_dicom_zip_file)],
        result_image_name="result_image_name",
        use_gpu=False,
        use_dicom=True)

    with mock.patch('score.init_from_model_inference_json',
                    return_value=(1, 2)):
        with mock.patch('score.run_inference',
                        return_value=True):
            with mock.patch('score.store_as_ubyte_nifti',
                            return_value=hnsegmentation_path):
                segmentation = score_image(score_pipeline_config)
                assert segmentation

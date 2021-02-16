#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import List
from unittest import mock
import zipfile
import numpy as np
import pytest
from pytorch_lightning import seed_everything

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint
from passthrough_model import PassThroughModel
from score import create_inference_pipeline, is_spacing_valid, run_inference, score_image, ScorePipelineConfig, \
    extract_zipped_dicom_series, convert_zipped_dicom_to_nifti, \
    convert_nifti_to_zipped_dicom_rt


test_image = full_ml_test_data_path("train_and_test_data") / "id1_channel1.nii.gz"
img_nii_path = full_ml_test_data_path("test_img.nii.gz")


# The directory containing this file.
THIS_DIR: Path = Path(__file__).parent.resolve()
# The TestData directory.
TEST_DATA_DIR: Path = THIS_DIR / "TestData"
# Filenames of dcm files in flat test zip.
TEST_FLAT_ZIP_FILENAMES = ['2.dcm', '3.dcm', '4.dcm']
# Flat test zip file.
TEST_FLAT_ZIP_FILE: Path = TEST_DATA_DIR / "test_flat.zip"
# As test_flat but everything in "folder1"
TEST_FLAT_NESTED_ZIP_FILE: Path = TEST_DATA_DIR / "test_flat_nested.zip"
# As test_flat_nested but everything in "folder2"
TEST_FLAT_NESTED_TWICE_ZIP_FILE: Path = TEST_DATA_DIR / "test_flat_nested_twice.zip"
# Filenames of dcm files in test two zip.
TEST_TWO_ZIP_FILENAMES = ['2.dcm', '3.dcm', '4.dcm', '6.dcm', '7.dcm', '8.dcm']
# Two folders containing dcm files
TEST_TWO_ZIP_FILE: Path = TEST_DATA_DIR / "test_two.zip"
# Two folders each containing a folder containing dcm files
TEST_TWO_NESTED_ZIP_FILE: Path = TEST_DATA_DIR / "test_two_nested.zip"
# As above, but all in another folder.
TEST_TWO_NESTED_TWICE_ZIP_FILE: Path = TEST_DATA_DIR / "test_two_nested_twice.zip"

# A sample H&N segmentation
HNSEGMENTATION_FILE = TEST_DATA_DIR / "hnsegmentation.nii.gz"
# A sample H&N DICOM series, zipped
HN_DICOM_SERIES_ZIP = TEST_DATA_DIR / "HN.zip"
# Expected zipped DICOM-RT file contents
HN_DICOM_RT_ZIPPED = ["hnsegmentation.nii.dcm"]


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


@pytest.mark.parametrize("zip_filename", [TEST_FLAT_ZIP_FILE, TEST_FLAT_NESTED_ZIP_FILE, TEST_FLAT_NESTED_TWICE_ZIP_FILE])
def test_unpack_flat_zip(zip_filename: Path, test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a zip file containing just files: 1.txt, 2.dcm, 3.dcm, 4.dcm and 5.txt
    can be extracted into a folder containing only the .dcm files.

    :param zip_filename: Path to test zip file.
    :param test_output_dirs: Test output directories.
    """
    _common_test_unpack_dicom_zip(zip_filename, TEST_FLAT_ZIP_FILENAMES, test_output_dirs)


@pytest.mark.parametrize("zip_filename", [TEST_TWO_ZIP_FILE, TEST_TWO_NESTED_ZIP_FILE, TEST_TWO_NESTED_TWICE_ZIP_FILE])
def test_unpack_two_zip(zip_filename: Path, test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a zip file containing files: 1.txt, 2.dcm, 3.dcm, 4.dcm and 5.txt in one folder
    and 5.txt, 6.dcm, 7.dcm, 8.dcm, 9.txt in another folder
    can be extracted into a folder containing only the .dcm files.

    :param zip_filename: Path to test zip file.
    :param test_output_dirs: Test output directories.
    """
    _common_test_unpack_dicom_zip(zip_filename, TEST_TWO_ZIP_FILENAMES, test_output_dirs)


def _common_test_unpack_dicom_zip(zip_filename: Path, expected_filenames: List[str],
                                  test_output_dirs: OutputFolderForTests) -> None:
    """
    Test the zip file contains expected .dcm files not in a folder.

    :param zip_filename: Path to test zip file.
    :param expected_filenames: List of expected filenames.
    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "unpack"
    extraction_folder = model_folder / "temp_zipped_dicom_extraction"
    extract_zipped_dicom_series(zip_filename, extraction_folder)
    _common_test_unpack_zip(extraction_folder, expected_filenames)


def _common_test_unpack_zip(extraction_folder: Path, expected_filenames: List[str]) -> None:
    """
    Test the extraction folder contains expected files, not in a folder.

    :param extraction_folder: Path to extracted zip file.
    :param expected_filenames: List of expected filenames.
    """
    # Get all files/folders in this series
    extracted_files = list(extraction_folder.glob('**/*'))
    # Check they are all files (no folders) and in the extraction_folder (not a subdirectory)
    for extracted_file in extracted_files:
        assert extracted_file.is_file()
        relative_path = extracted_file.relative_to(extraction_folder)
        assert str(relative_path.parent) == '.'
    extracted_file_names = sorted([extracted_file.name for extracted_file in extracted_files])
    # Check names are as expected
    assert extracted_file_names == expected_filenames


def assert_zip_file_contents(zip_filename: Path, expected_filenames: List[str],
                             model_folder: Path) -> None:
    """
    Compare the contents of a zip file, with a list of expected filenames.

    :param zip_filename: Path to zip file.
    :param expected_filenames: List of expected filenames.
    :param model_folder: Scratch folder.
    """
    assert zip_filename.is_file()
    extraction_folder = model_folder / "temp_zip_extraction"
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        zip_file.extractall(extraction_folder)
    _common_test_unpack_zip(extraction_folder, expected_filenames)


def test_convert_nifti_to_zipped_dicom_rt(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test calling convert_nifti_to_zipped_dicom_rt.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"
    model_folder.mkdir()

    nifti_filename, reference_series_folder = convert_zipped_dicom_to_nifti(HN_DICOM_SERIES_ZIP,
                                                                            model_folder)
    model_config = PassThroughModel()
    result_dst = convert_nifti_to_zipped_dicom_rt(HNSEGMENTATION_FILE, reference_series_folder, model_folder,
                                                  model_config)
    assert_zip_file_contents(result_dst, HN_DICOM_RT_ZIPPED, model_folder)


def test_score_image_dicom_two_inputs(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in with more than one input fails.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HN_DICOM_SERIES_ZIP), str(HN_DICOM_SERIES_ZIP)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    with pytest.raises(ValueError):
        score_image(score_pipeline_config)


def test_score_image_dicom_not_zip_input(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in with input not a zip file fails.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HNSEGMENTATION_FILE)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    with pytest.raises(zipfile.BadZipFile):
        score_image(score_pipeline_config)


def test_score_image_dicom_mock_all(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works, by mocking out functions that do most of the work.

    :param test_output_dirs: Test output directories.
    """
    mock_pipeline_base = {'mock_pipeline_base': True}
    model_config = PassThroughModel()
    mock_segmentation = {'mock_segmentation': True}

    model_folder = test_output_dirs.root_dir / "final"
    model_folder.mkdir()

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HN_DICOM_SERIES_ZIP)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    with mock.patch('score.init_from_model_inference_json',
                    return_value=(mock_pipeline_base, model_config)) as mock_init_from_model_inference_json:
        with mock.patch('score.run_inference',
                        return_value=mock_segmentation) as mock_run_inference:
            with mock.patch('score.store_as_ubyte_nifti',
                            return_value=HNSEGMENTATION_FILE) as mock_store_as_ubyte_nifti:
                segmentation = score_image(score_pipeline_config)
                assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

    mock_init_from_model_inference_json.assert_called_once_with(Path(score_pipeline_config.model_folder),
                                                                score_pipeline_config.use_gpu)
    mock_run_inference.assert_called()
    mock_store_as_ubyte_nifti.assert_called()


def test_score_image_dicom_mock_run_store(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works, by mocking out run and store functions.

    :param test_output_dirs: Test output directories.
    """
    mock_segmentation = {'mock_segmentation': True}
    model_config = DummyModel()
    model_config.set_output_to(test_output_dirs.root_dir)
    checkpoint_path = model_config.checkpoint_folder / "checkpoint.ckpt"
    create_model_and_store_checkpoint(model_config, checkpoint_path)

    azure_config = AzureConfig()
    project_root = Path(__file__).parent.parent
    ml_runner = MLRunner(model_config=model_config, azure_config=azure_config, project_root=project_root)
    model_folder = test_output_dirs.root_dir / "final"
    ml_runner.copy_child_paths_to_folder(model_folder=model_folder, checkpoint_paths=[checkpoint_path])

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HN_DICOM_SERIES_ZIP)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    with mock.patch('score.run_inference',
                    return_value=mock_segmentation) as mock_run_inference:
        with mock.patch('score.store_as_ubyte_nifti',
                        return_value=HNSEGMENTATION_FILE) as mock_store_as_ubyte_nifti:
            segmentation = score_image(score_pipeline_config)
            assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

    mock_run_inference.assert_called()
    mock_store_as_ubyte_nifti.assert_called()


def test_score_image_dicom_mock_run(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works, by mocking out only the run scoring function.

    :param test_output_dirs: Test output directories.
    """
    model_config = DummyModel()
    model_config.set_output_to(test_output_dirs.root_dir)
    checkpoint_path = model_config.checkpoint_folder / "checkpoint.ckpt"
    create_model_and_store_checkpoint(model_config, checkpoint_path)

    azure_config = AzureConfig()
    project_root = Path(__file__).parent.parent
    ml_runner = MLRunner(model_config=model_config, azure_config=azure_config, project_root=project_root)
    model_folder = test_output_dirs.root_dir / "final"
    ml_runner.copy_child_paths_to_folder(model_folder=model_folder, checkpoint_paths=[checkpoint_path])

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HN_DICOM_SERIES_ZIP)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    image_with_header = io_util.load_nifti_image(HNSEGMENTATION_FILE)

    with mock.patch('score.run_inference',
                    return_value=image_with_header.image) as mock_run_inference:
        segmentation = score_image(score_pipeline_config)
        assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

    mock_run_inference.assert_called()


def test_score_image_dicom_mock_none(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works.

    :param test_output_dirs: Test output directories.
    """
    model_config = PassThroughModel()
    model_config.set_output_to(test_output_dirs.root_dir)
    checkpoint_path = model_config.checkpoint_folder / "checkpoint.ckpt"
    create_model_and_store_checkpoint(model_config, checkpoint_path)

    azure_config = AzureConfig()
    project_root = Path(__file__).parent.parent
    ml_runner = MLRunner(model_config=model_config, azure_config=azure_config, project_root=project_root)
    model_folder = test_output_dirs.root_dir / "final"
    ml_runner.copy_child_paths_to_folder(model_folder=model_folder, checkpoint_paths=[checkpoint_path])

    score_pipeline_config = ScorePipelineConfig(
        data_folder=TEST_DATA_DIR,
        model_folder=str(model_folder),
        image_files=[str(HN_DICOM_SERIES_ZIP)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True)

    segmentation = score_image(score_pipeline_config)
    assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

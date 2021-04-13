#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
import shutil
from typing import List
from unittest import mock
import zipfile
import numpy as np
import pytest
from pytorch_lightning import seed_everything

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_ZIP_DICOM_NAME
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.configs.unit_testing.passthrough_model import PassThroughModel
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import reverse_tuple_float3
from Tests.ML.configs.DummyModel import DummyModel
from Tests.ML.utils.test_io_util import HNSEGMENTATION_FILE, zip_known_dicom_series

from Tests.ML.utils.test_model_util import create_model_and_store_checkpoint
from score import create_inference_pipeline, is_spacing_valid, run_inference, score_image, ScorePipelineConfig, \
    extract_zipped_files_and_flatten, convert_zipped_dicom_to_nifti, \
    convert_nifti_to_zipped_dicom_rt

test_image = full_ml_test_data_path("train_and_test_data") / "id1_channel1.nii.gz"
img_nii_path = full_ml_test_data_path("test_img.nii.gz")
# Expected zipped DICOM-RT file contents, just DEFAULT_RESULT_ZIP_DICOM_NAME without the final suffix.
HN_DICOM_RT_ZIPPED = ["segmentation.dcm"]


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


# First set of dummy filenames for testing zips.
TEST_ZIP_FILENAMES_1: List[str] = [f"{i}.txt" for i in range(1, 6)]
# Second set of dummy filenames for testing zips, distinct from TEST_ZIP_FILENAMES_1.
TEST_ZIP_FILENAMES_2: List[str] = [f"{i}.txt" for i in range(6, 11)]
# Third set of dummy filenames for testing zips, distinct again.
TEST_ZIP_FILENAMES_3: List[str] = [f"{i}.txt" for i in range(11, 13)]

# Test set for all files in a single folder.
TEST_ZIP_FILE_PATHS_ALL: List[List[Path]] = [
    # Simplest test zip file, just a list of files.
    [Path(f) for f in TEST_ZIP_FILENAMES_1],
    # As above but everything in "folder1"
    [Path("folder1") / f for f in TEST_ZIP_FILENAMES_1],
    # As above but everything in "folder2"
    [Path("folder2") / "folder1" / f for f in TEST_ZIP_FILENAMES_1]
]


@pytest.mark.parametrize("zip_file_contents", TEST_ZIP_FILE_PATHS_ALL)
def test_unpack_one_set_zip(zip_file_contents: List[Path], test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a zip file containing a set of files in one folder, but possibly in a series of nesting folders,
    can be extracted into a folder containing only the files.

    :param zip_file_contents: List of relative file paths to create and test.
    :param test_output_dirs: Test output directories.
    """
    _common_test_unpack_dicom_zip(zip_file_contents, TEST_ZIP_FILENAMES_1, test_output_dirs)


# Test set for all files in a two folders.
TEST_ZIP_FILE_PATHS_ALL2: List[List[Path]] = [
    # Two folders containing dcm files
    [Path("folder1") / f for f in TEST_ZIP_FILENAMES_1] + \
    [Path("folder3") / f for f in TEST_ZIP_FILENAMES_2],
    # Two folders each containing a folder containing dcm files
    [Path("folder2") / "folder1" / f for f in TEST_ZIP_FILENAMES_1] + \
    [Path("folder4") / "folder3" / f for f in TEST_ZIP_FILENAMES_2],
    # As above, but all in another folder.
    [Path("folder5") / "folder2" / "folder1" / f for f in TEST_ZIP_FILENAMES_1] + \
    [Path("folder5") / "folder3" / f for f in TEST_ZIP_FILENAMES_2]
]


@pytest.mark.parametrize("zip_file_contents", TEST_ZIP_FILE_PATHS_ALL2)
def test_unpack_two_distinct_sets_zip(zip_file_contents: List[Path], test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a zip file containing two distinct set of files in two folders, but possibly in a series of nesting
    folders,
    can be extracted into a folder containing only the files.

    :param zip_file_contents: List of relative file paths to create and test.
    :param test_output_dirs: Test output directories.
    """
    all_zip_filenames = TEST_ZIP_FILENAMES_1 + TEST_ZIP_FILENAMES_2
    _common_test_unpack_dicom_zip(zip_file_contents, all_zip_filenames, test_output_dirs)


# Test set for all files in a two folders with duplicates.
TEST_ZIP_FILE_PATHS_ALL3: List[List[Path]] = [
    # Two folders containing dcm files
    [Path("folder1") / f for f in TEST_ZIP_FILENAMES_1 + TEST_ZIP_FILENAMES_3] + \
    [Path("folder3") / f for f in TEST_ZIP_FILENAMES_2 + TEST_ZIP_FILENAMES_3],
    # Two folders each containing a folder containing dcm files
    [Path("folder2") / "folder1" / f for f in TEST_ZIP_FILENAMES_1 + TEST_ZIP_FILENAMES_3] + \
    [Path("folder4") / "folder3" / f for f in TEST_ZIP_FILENAMES_2 + TEST_ZIP_FILENAMES_3],
    # As above, but all in another folder.
    [Path("folder5") / "folder2" / "folder1" / f for f in TEST_ZIP_FILENAMES_1 + TEST_ZIP_FILENAMES_3] + \
    [Path("folder5") / "folder3" / f for f in TEST_ZIP_FILENAMES_2 + TEST_ZIP_FILENAMES_3]
]


@pytest.mark.parametrize("zip_file_contents", TEST_ZIP_FILE_PATHS_ALL3)
def test_unpack_two_overlapping_sets_zip(zip_file_contents: List[Path], test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that a zip file containing two set of files in two folders, but possibly in a series of nesting folders,
    cannot be extracted if the sets are not distinct.

    :param zip_file_contents: List of relative file paths to create and test.
    :param test_output_dirs: Test output directories.
    """
    all_zip_filenames = list(set(TEST_ZIP_FILENAMES_1 + TEST_ZIP_FILENAMES_2 + TEST_ZIP_FILENAMES_3))
    with pytest.raises(ValueError) as e:
        _common_test_unpack_dicom_zip(zip_file_contents, all_zip_filenames, test_output_dirs)
    assert str(e.value).startswith("Zip file contains duplicates.\n")


def _common_test_unpack_dicom_zip(zip_file_contents: List[Path], expected_filenames: List[str],
                                  test_output_dirs: OutputFolderForTests) -> None:
    """
    Test the zip file contains expected .dcm files not in a folder.

    :param zip_file_contents: List of relative file paths to create and test.
    :param expected_filenames: List of expected filenames.
    :param test_output_dirs: Test output directories.
    """
    pack_folder = test_output_dirs.root_dir / "pack"
    for zip_file_item in zip_file_contents:
        dummy_file = pack_folder / zip_file_item
        dummy_file.parent.mkdir(parents=True, exist_ok=True)
        dummy_file.touch()

    zip_filename = test_output_dirs.root_dir / "test.zip"
    shutil.make_archive(str(zip_filename.with_suffix('')), 'zip', str(pack_folder))

    extraction_folder = test_output_dirs.root_dir / "unpack"
    extract_zipped_files_and_flatten(zip_filename, extraction_folder)
    assert_folder_contents(extraction_folder, expected_filenames)


def assert_folder_contents(folder: Path, expected_filenames: List[str]) -> None:
    """
    Test the folder contains only expected files and there are no subfolders.

    :param folder: Path to folder to test.
    :param expected_filenames: List of expected filenames.
    """
    # Get all files/folders in this series
    folder_files = list(folder.glob('**/*'))
    # Check they are all files (no folders) and in the folder (not a subdirectory)
    for folder_file in folder_files:
        assert folder_file.is_file()
        relative_path = folder_file.relative_to(folder)
        assert str(relative_path.parent) == '.'
    folder_file_names = [folder_file.name for folder_file in folder_files]
    # Check names are as expected
    assert sorted(folder_file_names) == sorted(expected_filenames)


def assert_zip_file_contents(zip_filename: Path, expected_filenames: List[str],
                             scratch_folder: Path) -> None:
    """
    Check that a zip file contains exactly expected_filenames and that the zip file
    has no folders.

    :param zip_filename: Path to zip file.
    :param expected_filenames: List of expected filenames.
    :param scratch_folder: Scratch folder.
    """
    assert zip_filename.is_file()
    extraction_folder = scratch_folder / "temp_zip_extraction"
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
        zip_file.extractall(extraction_folder)
    assert_folder_contents(extraction_folder, expected_filenames)


def test_convert_nifti_to_zipped_dicom_rt(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test calling convert_nifti_to_zipped_dicom_rt.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"
    model_folder.mkdir()

    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)
    reference_series_folder = model_folder / "temp_extraction"
    nifti_filename = model_folder / "temp_nifti.nii.gz"
    convert_zipped_dicom_to_nifti(zipped_dicom_series_path, reference_series_folder, nifti_filename)
    model_config = PassThroughModel()
    result_dst = convert_nifti_to_zipped_dicom_rt(HNSEGMENTATION_FILE, reference_series_folder, model_folder,
                                                  model_config, DEFAULT_RESULT_ZIP_DICOM_NAME, model_id="test_model:1")
    assert_zip_file_contents(result_dst, HN_DICOM_RT_ZIPPED, model_folder)


def test_score_image_dicom_two_inputs(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that scoring with use_dicom and more than one input raises an exception.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"
    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)

    score_pipeline_config = ScorePipelineConfig(
        data_folder=zipped_dicom_series_path.parent,
        model_folder=str(model_folder),
        image_files=[str(zipped_dicom_series_path), str(zipped_dicom_series_path)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

    with pytest.raises(ValueError) as e:
        score_image(score_pipeline_config)
    assert str(e.value) == "Supply exactly one zip file in args.images."


def test_score_image_dicom_not_zip_input(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in with input not a zip file fails.

    :param test_output_dirs: Test output directories.
    """
    model_folder = test_output_dirs.root_dir / "final"
    model_folder.mkdir()
    test_file = model_folder / "test.txt"
    with test_file.open('w') as f:
        f.write("")

    score_pipeline_config = ScorePipelineConfig(
        data_folder=model_folder,
        model_folder=str(model_folder),
        image_files=[str(test_file)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

    with pytest.raises(zipfile.BadZipFile):
        score_image(score_pipeline_config)


def test_score_image_dicom_mock_all(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works, by mocking out functions that do most of the work.

    This mocks out init_from_model_inference_json, run_inference and store_as_ubyte_nifti so that
    only the skeleton of the logic is tested, particularly the final conversion to DICOM-RT.

    :param test_output_dirs: Test output directories.
    """
    mock_pipeline_base = {'mock_pipeline_base': True}
    model_config = PassThroughModel()
    mock_segmentation = {'mock_segmentation': True}

    model_folder = test_output_dirs.root_dir / "final"
    model_folder.mkdir()

    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)

    score_pipeline_config = ScorePipelineConfig(
        data_folder=zipped_dicom_series_path.parent,
        model_folder=str(model_folder),
        image_files=[str(zipped_dicom_series_path)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

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

    This mocks out run_inference and store_as_ubyte_nifti so that init_from_model_inference_json
    is tested in addition to the tests in test_score_image_dicom_mock_all.

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

    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)

    score_pipeline_config = ScorePipelineConfig(
        data_folder=zipped_dicom_series_path.parent,
        model_folder=str(model_folder),
        image_files=[str(zipped_dicom_series_path)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

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

    This mocks out run_inference so that store_as_ubyte_nifti
    is tested in addition to the tests in test_score_image_dicom_mock_run_store.

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

    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)

    score_pipeline_config = ScorePipelineConfig(
        data_folder=zipped_dicom_series_path.parent,
        model_folder=str(model_folder),
        image_files=[str(zipped_dicom_series_path)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

    image_with_header = io_util.load_nifti_image(HNSEGMENTATION_FILE)

    with mock.patch('score.run_inference',
                    return_value=image_with_header.image) as mock_run_inference:
        segmentation = score_image(score_pipeline_config)
        assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

    mock_run_inference.assert_called()


def test_score_image_dicom_mock_none(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that dicom in and dicom-rt out works.

    Finally there is no mocking and full image scoring is run using the PassThroughModel.

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

    zipped_dicom_series_path = test_output_dirs.root_dir / "temp_pack_dicom_series" / "dicom_series.zip"
    zip_known_dicom_series(zipped_dicom_series_path)

    score_pipeline_config = ScorePipelineConfig(
        data_folder=zipped_dicom_series_path.parent,
        model_folder=str(model_folder),
        image_files=[str(zipped_dicom_series_path)],
        result_image_name=HNSEGMENTATION_FILE.name,
        use_gpu=False,
        use_dicom=True,
        model_id="Dummy:1")

    segmentation = score_image(score_pipeline_config)
    assert_zip_file_contents(segmentation, HN_DICOM_RT_ZIPPED, model_folder)

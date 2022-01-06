import os

import pandas as pd

from InnerEye.Common.fixed_paths_for_tests import tests_root_directory
from InnerEye.ML.Histopathology.datasets.base_dataset import SlidesDataset
from InnerEye.ML.Histopathology.utils.naming import SlideKey

HISTO_TEST_DATA_DIR = str(tests_root_directory("ML/histopathology/test_data"))


class MockSlidesDataset(SlidesDataset):
    DEFAULT_CSV_FILENAME = "test_slides_dataset.csv"
    METADATA_COLUMNS = ('meta1', 'meta2')

    def __init__(self) -> None:
        super().__init__(root=HISTO_TEST_DATA_DIR)


def test_slides_dataset() -> None:
    dataset = MockSlidesDataset()
    assert isinstance(dataset.dataset_df, pd.DataFrame)
    assert dataset.dataset_df.index.name == dataset.SLIDE_ID_COLUMN
    assert len(dataset) == len(dataset.dataset_df)

    sample = dataset[0]
    assert isinstance(sample, dict)
    assert all(isinstance(key, SlideKey) for key in sample)

    expected_keys = [SlideKey.SLIDE_ID, SlideKey.IMAGE, SlideKey.IMAGE_PATH, SlideKey.LABEL,
                     SlideKey.METADATA]
    assert all(key in sample for key in expected_keys)

    image_path = sample[SlideKey.IMAGE_PATH]
    assert isinstance(image_path, str)
    assert os.path.isfile(image_path)

    metadata = sample[SlideKey.METADATA]
    assert isinstance(metadata, dict)
    assert all(meta_col in metadata for meta_col in type(dataset).METADATA_COLUMNS)

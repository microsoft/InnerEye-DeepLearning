#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import pytest

from InnerEye.ML.utils.csv_util import CSV_DATE_HEADER, CSV_FEATURE_HEADER, CSV_PATH_HEADER, CSV_SUBJECT_HEADER, \
    OutlierType, drop_rows_missing_important_values, extract_outliers, load_csv
from Tests.fixed_paths_for_tests import full_ml_test_data_path

known_csv_path = full_ml_test_data_path("hdf5_data") / "dataset.csv"
nonexistent_csv_path = full_ml_test_data_path("hdf5_data") / "idontexist.csv"
known_csv_missing_vals_path = full_ml_test_data_path("hdf5_data") / "dataset_missing_values.csv"
known_df_cols = [CSV_PATH_HEADER, CSV_FEATURE_HEADER, CSV_DATE_HEADER, CSV_SUBJECT_HEADER]


@pytest.mark.parametrize('csv_path', [known_csv_path])
def test_load_csv(csv_path: Path) -> None:
    """
    Check that loaded dataframe has the expected number of rows.
    """
    df = load_csv(csv_path, expected_cols=known_df_cols)
    assert len(df) == 6
    assert all([x in list(df.columns) for x in known_df_cols])


def test_load_nonexistent_csv() -> None:
    """
    Check that an error is returned when attempting to load a non-existent CSV.
    """
    expected_cols = [CSV_PATH_HEADER, CSV_FEATURE_HEADER]
    with pytest.raises(Exception) as exc:
        load_csv(nonexistent_csv_path, expected_cols)
    assert str(exc.value) == "No CSV file exists at this location: {0}".format(nonexistent_csv_path)


def test_load_csv_no_expected_cols() -> None:
    """
    Check that an error is raised when the user neglects to provide a list of expected columns.
    """
    with pytest.raises(Exception):
        load_csv(known_csv_path, [])


def test_drop_rows_missing_important_values() -> None:
    """
    Test that rows missing important values are dropped from the DataFrame.
    """
    df = pd.read_csv(known_csv_missing_vals_path)
    assert (len(df) == 6)
    important_cols = [CSV_PATH_HEADER, CSV_FEATURE_HEADER]
    df = drop_rows_missing_important_values(df, important_cols)
    assert len(df) == 3


def test_extract_outliers() -> None:
    """
    Test that extract_outliers correctly returns the DataFrame rows where Dice < mean - outlier_range * std
    """
    test_df = pd.DataFrame({
        "Dice": range(10)
    })

    # check the outliers are expected 0, 1 and 2 deviations less than the mean
    assert list(range(5)) == list(extract_outliers(test_df, 0).Dice.values)
    assert list(range(2)) == list(extract_outliers(test_df, 1).Dice.values)
    assert list() == list(extract_outliers(test_df, 2).Dice.values)


def test_extract_outliers_higher() -> None:
    """
    Test that extract_outliers correctly returns the DataFrame rows where
    Hausdorff distance > mean + outlier_range * std
    """
    test_df = pd.DataFrame({
        "Hausdorff": range(10)
    })
    assert list(range(5, 10, 1)) == list(extract_outliers(test_df, 0, "Hausdorff",
                                                          outlier_type=OutlierType.HIGH).Hausdorff.values)
    assert list(range(8, 10, 1)) == list(extract_outliers(test_df, 1, "Hausdorff",
                                                          outlier_type=OutlierType.HIGH).Hausdorff.values)
    assert list() == list(extract_outliers(test_df, 2, "Hausdorff", outlier_type=OutlierType.HIGH).Hausdorff.values)

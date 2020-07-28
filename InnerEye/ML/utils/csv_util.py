#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from InnerEye.ML.utils.metrics_constants import MetricsFileColumns

CSV_FEATURE_HEADER: str = "feature"
CSV_DATE_HEADER: str = "acquisition_date"
CSV_SUBJECT_HEADER: str = "subject"
CSV_PATH_HEADER: str = "filePath"
CSV_CHANNEL_HEADER: str = "channel"
CSV_INSTITUTION_HEADER: str = "institutionId"
CSV_SERIES_HEADER: str = "seriesId"
CSV_TAGS_HEADER: str = "tags"

COL_DICE = MetricsFileColumns.Dice.value
COL_SPLIT = "split"


class OutlierType(Enum):
    HIGH = "High"
    LOW = "Low"


def load_csv(csv_path: Path, expected_cols: List[str], col_type_converters: Optional[Dict[str, Any]] = None
             ) -> pd.DataFrame:
    """
    Load a pandas dataframe from a csv. If the columns do not contain at least expected_cols, an exception is raised

    :param csv_path: Path to file
    :param expected_cols: A list of the columns which must, as a minimum, be present.
    :param col_type_converters: Dictionary of column: type, which ensures certain DataFrame columns are parsed with
    specific types
    :return: Loaded pandas DataFrame
    """
    if not expected_cols:
        raise ValueError("You must provide a list of at least one of the expected column headings of your CSV.")
    if not csv_path.is_file():
        raise FileNotFoundError("No CSV file exists at this location: {0}".format(csv_path))

    df = pd.read_csv(csv_path, converters=col_type_converters)
    if len(df) == 0:
        raise ValueError("Dataset at {0} contains no values".format(csv_path))
    # Check that all of the expected column headers are present in the CSV.
    actual_cols = list(df.columns)
    if not set(expected_cols).issubset(actual_cols):
        raise ValueError("CSV should at least contain the columns {0} but found {1}".format(expected_cols, actual_cols))
    return df


def drop_rows_missing_important_values(df: pd.DataFrame, important_cols: List[str]) -> pd.DataFrame:
    """
    Remove rows from the DataFrame in which the columns that have been specified by the user as "important" contain
    null values or only whitespace.
    :param df: DataFrame
    :param important_cols: Columns which must not contain null values
    :return: df: DataFrame without the dropped rows.
    """
    df = df.replace(r'^\s*$', np.nan, regex=True)
    before_len = len(df)
    df = df.dropna(subset=important_cols)
    num_dropped = len(df) - before_len
    if num_dropped > 0:
        logging.info("Dropping {0} rows from the data set since they are missing values from one of the columns: {1}"
                     .format(num_dropped, important_cols))
    return df


def extract_outliers(df: pd.DataFrame, outlier_range: float, outlier_col: str = COL_DICE,
                     outlier_type: OutlierType = OutlierType.LOW) -> pd.DataFrame:
    """
    Given a DataFrame, extract the subset in which a given value (specified by outlier_col) falls outside of
    mean +- outlier_range * std.

    :param df: DataFrame from which to extract the outliers
    :param outlier_range: The number of standard deviation from the mean which the points have to be apart
    to be considered an outlier i.e. a point is considered an outlier if its outlier_col value is above
    mean + outlier_range * std (if outlier_type is HIGH) or below mean - outlier_range * std (if outlier_type is
    LOW).
    :param outlier_col: The column from which to calculate outliers, e.g. Dice
    :param outlier_type: Either LOW (i.e. below accepted range) or HIGH (above accepted range) outliers.
    :return: DataFrame containing only the outliers
    """
    if outlier_range < 0:
        raise ValueError("outlier_range must be non-negative. Found: {}".format(outlier_range))
    if outlier_type == OutlierType.LOW:
        return df[df[outlier_col] < df[outlier_col].mean() - outlier_range * df[outlier_col].std()]
    elif outlier_type == OutlierType.HIGH:
        return df[df[outlier_col] > df[outlier_col].mean() + outlier_range * df[outlier_col].std()]
    raise ValueError(f"Outlier type must be one of LOW or HIGH. Received {outlier_type}")


def get_worst_performing_outliers(df: pd.DataFrame,
                                  outlier_range: float,
                                  outlier_col_name: str = COL_DICE,
                                  max_n_outliers: int = None) -> List[Tuple[int, str, float, str]]:
    """
    Returns a sorted list (worst to best) of all the worst performing outliers in the metrics table
    according to metric provided by outlier_col_name
    :param df: Metrics DataFrame
    :param outlier_col_name: The column by which to determine outliers
    :param outlier_range: The standard deviation from the mean which the points have to be below
    to be considered an outlier.
    :param max_n_outliers: the number of (worst performing) outlier IDs to return.
    :return: a sorted list (worst to best) of all the worst performing outliers
    """
    if outlier_col_name not in df.columns:
        raise ValueError(f"Column {outlier_col_name} is not present in DataFrame columns: {df.columns.tolist()}")

    outlier_df = extract_outliers(df, outlier_range, outlier_col=outlier_col_name).drop([COL_SPLIT], axis=1)
    sorted_outlier_df = outlier_df.sort_values(by=outlier_col_name, ascending=True)

    ids_and_structures = list(zip(sorted_outlier_df.Patient.values.astype(int),
                                  sorted_outlier_df.Structure.values.astype(str),
                                  sorted_outlier_df[outlier_col_name].values.astype(float),
                                  sorted_outlier_df.seriesId.values.astype(str)))

    if max_n_outliers is not None:
        return ids_and_structures[:max_n_outliers]
    return ids_and_structures

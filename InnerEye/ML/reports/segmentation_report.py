#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from pandas import DataFrame

from InnerEye.ML.reports.notebook_report import print_header
from InnerEye.ML.utils.csv_util import OutlierType, extract_outliers
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns


def plot_scores_for_csv(path_csv: str, outlier_range: float) -> None:
    """
    Displays all the tables and figures given a csv file with segmentation metrics
    Columns expected: Patient,Structure,Dice,HausdorffDistance_mm,MeanDistance_mm
    """
    print(f"Reading raw metrics data from: {path_csv}")
    df = pd.read_csv(path_csv)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
        display_metric(df, MetricsFileColumns.Dice.value, outlier_range, high_values_are_good=True)
        display_metric(df, MetricsFileColumns.HausdorffDistanceMM.value, outlier_range, high_values_are_good=False)
        display_metric(df, MetricsFileColumns.MeanDistanceMM.value, outlier_range, high_values_are_good=False)


def display_metric(df: pd.DataFrame, metric_name: str, outlier_range: float, high_values_are_good: bool) -> None:
    """
    Displays a dataframe with a metric per structure, first showing the
    :param df: The dataframe with metrics per structure
    :param metric_name: The metric to sort by.
    :param outlier_range: The standard deviation range data points must fall outside of to be considered an outlier
    :param high_values_are_good: If true, high values of the metric indicate good performance. If false, low
    values indicate good performance.
    """
    print_header(metric_name, level=2)
    # Display with best structures first.
    display(describe_score(df, metric_name, ascending=(not high_values_are_good)))
    print_header(f"Worst patients by {metric_name} (< {outlier_range} sd)", level=3)
    display(display_outliers(df, outlier_range, metric_name, high_values_are_good))
    boxplot_per_structure(df, column_name=metric_name, title=metric_name)
    plt.show()


def display_outliers(df: pd.DataFrame, outlier_range: float,
                     metric_name: str, high_values_are_good: bool) -> pd.DataFrame:
    """
    Prints a dataframe that contains the worst patients by the given metric.
    "Worst" is determined by having a metric value which is less than the outlier_range
    standard deviation from the mean.
    """
    return extract_outliers(df, outlier_range, metric_name,
                            OutlierType.LOW if high_values_are_good else OutlierType.HIGH)


def describe_score(df: pd.DataFrame, metric_name: str, ascending: bool = True) -> pd.DataFrame:
    """
    Creates a per-structure breakdown of the given metric, and prints quartiles, mean and median.
    The table is sorted by the mean value of the metric, either ascending or descending.
    """
    df2 = df.groupby(MetricsFileColumns.Structure.value)[metric_name].describe() \
        .unstack(1).reset_index()
    df2 = pd.pivot_table(df2, values=[0], index=MetricsFileColumns.Structure.value,
                         columns='level_0').reset_index()
    df2.columns = df2.columns.droplevel()
    df2 = df2.sort_values(by='mean', ascending=ascending).reset_index(drop=True)
    df2.columns = [MetricsFileColumns.Structure.value] + list(df2.columns.array)[1:]
    return df2


def boxplot_per_structure(df: DataFrame, column_name: str,
                          title: str) -> None:
    """
    Creates a box-and-whisker plot for a score per structure. Structures are on the x-axis,
    box plots are drawn vertically. The plot is created in the currently active figure or subplot.
    """
    structure = MetricsFileColumns.Structure.value
    dice_numeric = column_name
    structure_series = df[structure]
    unique_structures = structure_series.unique()
    dice_per_structure = [df[dice_numeric][structure_series == s] for s in unique_structures]
    # If there are only single entries per structure, do not generate a box plot
    if all([len(dps) == 1 for dps in dice_per_structure]):
        return

    plt.title(title)
    plt.boxplot(dice_per_structure, labels=unique_structures)
    plt.xlabel("Structure")
    plt.ylabel(column_name)
    plt.xticks(rotation=75)
    plt.grid()

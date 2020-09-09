#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns
from InnerEye.ML.utils.metrics_util import boxplot_per_structure

SEGMENTATION_REPORT_NOTEBOOK_PATH = Path(__file__).parent.absolute() / "segmentation_report.ipynb"
TEST_METRICS_CSV_PARAMETER_NAME = "test_metrics_csv"
TRAIN_METRICS_CSV_PARAMETER_NAME = "train_metrics_csv"
VAL_METRICS_CSV_PARAMETER_NAME = "val_metrics_csv"
INNEREYE_PATH_PARAMETER_NAME = "innereye_path"


def plot_scores_for_csv(path_csv: str):
    """
    Displays all the tables and figures given a csv file with segmentation metrics
    Columns expected: Patient,Structure,Dice,HausdorffDistance_mm,MeanDistance_mm
    """
    df = pd.read_csv(path_csv)

    DisplayMetric(df, MetricsFileColumns.Dice.value)
    DisplayMetric(df, MetricsFileColumns.HausdorffDistanceMM.value)
    DisplayMetric(df, MetricsFileColumns.MeanDistanceMM.value)


def DisplayMetric(df, metric_name: str):
    display(Markdown(f"##{metric_name}"))
    display(describe_score(df, MetricsFileColumns.Dice.value))
    display(Markdown(f"##Worse {metric_name} patients"))
    display(worse_patients(df, MetricsFileColumns.Dice.value, ascending=True))
    boxplot_per_structure(df, column_name="Dice", title="Dice")
    plt.show()


def worse_patients(df: pd.DataFrame, metric_name: str, ascending: bool):
    df2 = df.sort_values(by=metric_name, ascending=ascending).head(20)
    return df2


def describe_score(df: pd.DataFrame, metric_name: str):
    df2 = df.groupby(MetricsFileColumns.Structure.value)[metric_name].describe() \
        .unstack(1).reset_index()
    df2 = pd.pivot_table(df2, values=[0], index=MetricsFileColumns.Structure.value,
                         columns='level_0').reset_index()
    df2.columns = df2.columns.droplevel()
    df2 = df2.sort_values(by='mean').reset_index(drop=True)
    df2.columns = [MetricsFileColumns.Structure.value] + list(df2.columns.array)[1:]
    return df2

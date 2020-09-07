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

SEGMENTATION_REPORT_NOTEBOOK_PATH = current_dir = Path(__file__).parent.absolute() / "segmentation_report.ipynb"


def plot_scores_for_csv(path_csv: str):
    df = pd.read_csv(path_csv)

    display(Markdown("##Dice"))
    describe_score(df, MetricsFileColumns.Dice.value)
    boxplot_per_structure(df, column_name="Dice", title="Dice")
    plt.show()

    display(Markdown("##HausdorffDistance_mm"))
    describe_score(df, MetricsFileColumns.HausdorffDistanceMM.value)
    boxplot_per_structure(df, column_name="HausdorffDistance_mm", title="HausdorffDistance_mm")
    plt.show()

    display(Markdown("##MeanDistance_mm"))
    describe_score(df, MetricsFileColumns.MeanDistanceMM.value)
    boxplot_per_structure(df, column_name="MeanDistance_mm", title="MeanDistance_mm")
    plt.show()


def describe_score(df: pd.DataFrame, metric_name: str):
    df2 = df.groupby(MetricsFileColumns.Structure.value)[metric_name].describe() \
        .unstack(1).reset_index()
    df2 = pd.pivot_table(df2, values=[0], index=MetricsFileColumns.Structure.value,
                         columns='level_0').reset_index()
    df2.columns = df2.columns.droplevel()
    df2 = df2.sort_values(by='mean')
    display(df2)

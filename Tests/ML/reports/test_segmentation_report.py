#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from pathlib import Path

import pandas as pd

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.reports.notebook_report import generate_segmentation_notebook
from InnerEye.ML.reports.segmentation_report import describe_score, worst_patients_and_outliers
from InnerEye.ML.utils.csv_util import COL_IS_OUTLIER
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns


def test_generate_segmentation_report(test_output_dirs: OutputFolderForTests) -> None:
    reports_folder = Path(__file__).parent
    metrics_file = reports_folder / "metrics_hn.csv"
    current_dir = test_output_dirs.make_sub_dir("test_segmentation_report")
    result_file = current_dir / "report.ipynb"
    result_html = generate_segmentation_notebook(result_notebook=result_file,
                                                 test_metrics=metrics_file)
    assert result_file.is_file()
    assert result_html.is_file()
    assert result_html.suffix == ".html"
    # Check html contains the name of a key structure
    contents = result_html.read_text(encoding='utf-8')
    assert 'parotid_r' in contents


def test_describe_metric() -> None:
    data = """Patient,Structure,Dice,HausdorffDistance_mm,MeanDistance_mm
13,brainstem,0.775,9.101,1.455
16,brainstem,0.782,11.213,1.657
19,brainstem,0.791,5.370,1.378
14,brainstem,0.795,8.375,1.439
15,brainstem,0.797,6.244,1.419
13,very_long_structure_name,0.116,5.456,2.931
16,very_long_structure_name,0.143,5.370,3.008
19,very_long_structure_name,0.162,6.810,2.388
14,very_long_structure_name,0.164,6.515,2.802
15,very_long_structure_name,0.182,3.843,1.801
"""
    df = pd.read_csv(StringIO(data))
    df2 = describe_score(df, MetricsFileColumns.Dice.value, ascending=True)
    # Do full string matching on the results, to ensure that the describe function uses a wide option context
    # for Pandas, to avoid column breaks at col 80
    expected = """                  Structure    25%    50%    75%  count    max    mean    min       std
0  very_long_structure_name  0.143  0.162  0.164    5.0  0.182  0.1534  0.116  0.025056
1                 brainstem  0.782  0.791  0.795    5.0  0.797  0.7880  0.775  0.009274
"""
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 150):
        assert str(df2).splitlines() == expected.splitlines()


def test_worst_patients() -> None:
    data = """Patient,foo
A,0
C,100
D,101
E,102
F,200"""
    df = pd.read_csv(StringIO(data))
    assert df["foo"].std() < 80
    # Metric values are constructed such that A and F are more than 1 std away from the mean. With outlier_range
    # set to 1, we should get A and F back as outliers.
    worst = worst_patients_and_outliers(df, outlier_range=1, metric_name="foo", high_values_are_good=True,
                                        max_row_count=2)
    assert worst["Patient"].to_list() == ["A", "C"]
    assert worst[COL_IS_OUTLIER].to_list() == ["Yes", ""]
    worst = worst_patients_and_outliers(df, outlier_range=1, metric_name="foo", high_values_are_good=False,
                                        max_row_count=2)
    assert worst["Patient"].to_list() == ["F", "E"]
    assert worst[COL_IS_OUTLIER].to_list() == ["Yes", ""]

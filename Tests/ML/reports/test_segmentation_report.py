#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd

from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.reports.segmentation.segmentation_report import INNEREYE_PATH_PARAMETER_NAME, \
    SEGMENTATION_REPORT_NOTEBOOK_PATH, TEST_METRICS_CSV_PARAMETER_NAME, describe_score

from InnerEye.ML.reports.notebook_report import generate_notebook
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns


def test_generate_segmentation_report(test_output_dirs: TestOutputDirectories) -> None:
    current_dir = Path(test_output_dirs.make_sub_dir("test_segmentation_report"))
    metrics_path = Path(__file__).parent.absolute() / "metrics_hn.csv"
    generate_notebook(notebook_path=SEGMENTATION_REPORT_NOTEBOOK_PATH,
                      notebook_params={TEST_METRICS_CSV_PARAMETER_NAME: str(metrics_path),
                                       INNEREYE_PATH_PARAMETER_NAME: str(Path(__file__).parent.parent.parent.parent)},
                      result_path=current_dir / "report.ipynb")
    chk_file = Path(current_dir / "report.ipynb")
    assert chk_file.is_file()
    chk_file = Path(current_dir / "report.html")
    assert chk_file.is_file()
    # Check html contains the name of a key structure
    with open(chk_file, "r", encoding='utf-8') as f:
        assert 'parotid_r' in f.read()


def test_describe_metric() -> None:
    current_dir = Path(__file__).parent.absolute()
    metrics_path = current_dir / "metrics_hn.csv"
    df = pd.read_csv(metrics_path)
    df2 = describe_score(df, MetricsFileColumns.Dice.value)
    assert list(df2.columns.array) == ['Structure', '25%', '50%', '75%', 'count', 'max', 'mean', 'min', 'std']

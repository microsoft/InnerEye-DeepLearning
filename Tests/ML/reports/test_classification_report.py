#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from InnerEye.Common.output_directories import TestOutputDirectories
from InnerEye.ML.reports.notebook_report import generate_classification_notebook


def test_generate_classification_report(test_output_dirs: TestOutputDirectories) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"
    current_dir = Path(test_output_dirs.make_sub_dir("test_classification_report"))
    result_file = current_dir / "report.ipynb"
    result_html = generate_classification_notebook(result_notebook=result_file,
                                                   val_metrics=val_metrics_file,
                                                   test_metrics=test_metrics_file)
    assert result_file.is_file()
    assert result_html.is_file()
    assert result_html.suffix == ".html"

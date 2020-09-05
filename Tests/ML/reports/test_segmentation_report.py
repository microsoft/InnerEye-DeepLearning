#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

from InnerEye.ML.reports.segmentation.segmentation_report import SEGMENTATION_REPORT_NOTEBOOK_PATH

from InnerEye.ML.reports.notebook_report import generate_notebook


def test_generate_segmentation_report():
    current_dir = Path(__file__).parent.absolute()
    metrics_path = current_dir / "metrics_hn.csv"
    generate_notebook(notebook_path=SEGMENTATION_REPORT_NOTEBOOK_PATH,
                      notebook_params={"metrics_csv": str(metrics_path)},
                      result_path=current_dir / "report.ipynb")

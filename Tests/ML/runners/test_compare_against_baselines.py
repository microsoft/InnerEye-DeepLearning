#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import pytest

from InnerEye.Common import common_util
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.baselines_util import ComparisonBaseline, get_comparison_baselines, perform_score_comparisons
from Tests.Common.test_util import DEFAULT_RUN_RECOVERY_ID
from Tests.ML.util import get_default_azure_config


@pytest.mark.skipif(common_util.is_windows(), reason="Loading tk sometimes fails on Windows")
def test_perform_score_comparisons() -> None:
    dataset_df = pd.DataFrame()
    dataset_df['subject'] = list(range(10))
    dataset_df['seriesId'] = [f"s{i}" for i in range(10)]
    dataset_df['institutionId'] = ["xyz"] * 10
    metrics_df = pd.DataFrame()
    metrics_df['Patient'] = list(range(10))
    metrics_df['Structure'] = ['appendix'] * 10
    metrics_df['Dice'] = [0.5 + i * 0.02 for i in range(10)]
    comparison_metrics_df = pd.DataFrame()
    comparison_metrics_df['Patient'] = list(range(10))
    comparison_metrics_df['Structure'] = ['appendix'] * 10
    comparison_metrics_df['Dice'] = [0.51 + i * 0.02 for i in range(10)]
    comparison_name = "DefaultName"
    comparison_run_rec_id = "DefaultRunRecId"
    baseline = ComparisonBaseline(comparison_name, dataset_df, comparison_metrics_df, comparison_run_rec_id)
    result = perform_score_comparisons(dataset_df, metrics_df, [baseline])
    assert result.did_comparisons
    assert len(result.wilcoxon_lines) == 5
    assert result.wilcoxon_lines[0] == f"Run 1: {comparison_name}"
    assert result.wilcoxon_lines[1] == "Run 2: CURRENT"
    assert result.wilcoxon_lines[3].find("WORSE") > 0
    assert list(result.plots.keys()) == [f"{comparison_name}_vs_CURRENT"]


def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    azure_config = get_default_azure_config()
    comparison_name = "DefaultName"
    comparison_path = DEFAULT_RUN_RECOVERY_ID + "/outputs/epoch_002/Test"
    baselines = get_comparison_baselines(test_output_dirs.root_dir,
                                         azure_config, [(comparison_name, comparison_path)])
    assert len(baselines) == 1
    assert baselines[0].name == comparison_name

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import pytest

from InnerEye.Common import common_util
from InnerEye.Common.common_util import BEST_EPOCH_FOLDER_NAME
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.baselines_util import ComparisonBaseline, get_comparison_baselines, perform_score_comparisons
from InnerEye.ML.common import ModelExecutionMode
from Tests.ML.util import get_default_azure_config
from Tests.AfterTraining.test_after_training import get_most_recent_run_id


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


@pytest.mark.after_training_single_run
def test_get_comparison_data(test_output_dirs: OutputFolderForTests) -> None:
    azure_config = get_default_azure_config()
    comparison_name = "DefaultName"
    comparison_path = get_most_recent_run_id() + \
                      f"/{DEFAULT_AML_UPLOAD_DIR}/{BEST_EPOCH_FOLDER_NAME}/{ModelExecutionMode.TEST.value}"
    baselines = get_comparison_baselines(test_output_dirs.root_dir,
                                         azure_config, [(comparison_name, comparison_path)])
    assert len(baselines) == 1
    assert baselines[0].name == comparison_name

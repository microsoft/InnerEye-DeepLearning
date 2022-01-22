#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Script to find mean and standard deviation of desired metrics from cross validation child runs.
"""
import sys, os
import pandas as pd
from pathlib import Path

current_dir = Path(os.getcwd())
radiomics_root = current_dir
if (radiomics_root / "InnerEyePrivate").is_dir():
    radiomics_root_str = str(radiomics_root)
    if radiomics_root_str not in sys.path:
        print(f"Adding to sys.path: {radiomics_root_str}")
        sys.path.insert(0, radiomics_root_str)
        sys.path.insert(0, str(radiomics_root / "innereye-deeplearning"))
        sys.path.insert(0, str(radiomics_root / "innereye-deeplearning/hi-ml/hi-ml-azure/src"))
        sys.path.insert(0, str(radiomics_root / "innereye-deeplearning/hi-ml/hi-ml/src"))
        print(f"Sys path {sys.path}")

from health_azure import aggregate_hyperdrive_metrics, get_workspace
from InnerEye.Common import fixed_paths


def get_cross_validation_metrics_df(run_id: str) -> pd.DataFrame:
    """
    Function to aggregate the metric over cross-validation runs
    :param run_id: run id of the hyperdrive run containing child runs
    """
    aml_workspace = get_workspace()
    os.chdir(fixed_paths.repository_root_directory())
    df = aggregate_hyperdrive_metrics(run_id=run_id,
                                      child_run_arg_name="cross_validation_split_index",
                                      aml_workspace=aml_workspace)
    return df


if __name__ == "__main__":
    metrics_list = ['test/accuracy', 'test/auroc', 'test/f1score', 'test/precision', 'test/recall', 'test/macro_accuracy', 'test/weighted_accuracy']
    run_id = "hsharma_panda_tiles_ssl:HD_b5be4968-4896-4fc4-8d62-291ebe5c57c2"
    metrics_df = get_cross_validation_metrics_df(run_id=run_id)
    for metric in metrics_list:
        if metric in metrics_df.index.values:
            mean = metrics_df.loc[[metric]].mean(axis=1)[metric]
            std = metrics_df.loc[[metric]].std(axis=1)[metric]
            print(f"{metric}: {round(mean,4)} ± {round(std,4)}")
        else:
            print(f"Metric {metric} not found in the Hyperdrive run metrics for run id {run_id}.")

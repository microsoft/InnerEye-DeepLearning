#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Script to find mean and standard deviation of desired metrics from cross validation child runs.
"""
import os
import pandas as pd

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
    metrics_list = ['test/accuracy', 'test/auroc', 'test/f1score', 'test/precision', 'test/recall']
    run_id = "hsharma_features_viz:HD_eff4c009-2f9f-4c2c-94c6-c0c84944a412"
    metrics_df = get_cross_validation_metrics_df(run_id=run_id)
    for metric in metrics_list:
        if metric in metrics_df.index.values:
            mean = metrics_df.loc[[metric]].mean(axis=1)[metric]
            std = metrics_df.loc[[metric]].std(axis=1)[metric]
            print(f"{metric}: {round(mean,4)} ± {round(std,4)}")
        else:
            print(f"Metric {metric} not found in the Hyperdrive run metrics for run id {run_id}.")

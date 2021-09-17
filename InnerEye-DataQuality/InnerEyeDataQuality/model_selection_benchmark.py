#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from default_paths import MAIN_SIMULATION_DIR, MODEL_SELECTION_BENCHMARK_DIR, FIGURE_DIR
from InnerEyeDataQuality.datasets.cifar10_utils import get_cifar10_label_names
from InnerEyeDataQuality.deep_learning.model_inference import inference_ensemble
from InnerEyeDataQuality.deep_learning.utils import create_logger, load_model_config, load_selector_config
from InnerEyeDataQuality.selection.selectors.label_based import cross_entropy
from InnerEyeDataQuality.selection.simulation import DataCurationSimulator
from InnerEyeDataQuality.selection.simulation_statistics import get_ambiguous_sample_ids
from InnerEyeDataQuality.utils.dataset_utils import get_datasets
from InnerEyeDataQuality.utils.generic import create_folder
from InnerEyeDataQuality.utils.plot import plot_confusion_matrix


def get_rank(array: Union[np.ndarray, List]) -> int:
    """
    Returns the ranking of an array where the highest value has a rank of 1 and
    the lowest value has the highest rank.
    """
    return len(array) - np.argsort(array).argsort()


def main(args: argparse.Namespace) -> None:

    # Parameters
    number_of_runs = 1
    evaluate_on_ambiguous_samples = False

    # Create the evaluation dataset - Make sure that it's the same dataset for all configs
    assert isinstance(args.config, list)
    _, dataset = get_datasets(load_model_config(args.config[0]),
                              use_augmentation=False,
                              use_noisy_labels_for_validation=True,
                              use_fixed_labels=False if number_of_runs > 1 else True)

    # Choose a subset of the dataset
    if evaluate_on_ambiguous_samples:
        ind = get_ambiguous_sample_ids(dataset.label_counts, threshold=0.10)  # type: ignore
    else:
        ind = range(len(dataset))  # type: ignore

    # If specified load curated dataset labels
    curated_target_labels = dict()
    for cfg_path in args.curated_label_config if args.curated_label_config else list():
        cfg = load_selector_config(cfg_path)
        search_dir = MAIN_SIMULATION_DIR / cfg.selector.output_directory
        targets_list = list()
        for _f in search_dir.glob('**/*.hdf'):
            _label_counts = DataCurationSimulator.load_simulator_results(_f)
            _targets = np.argmax(_label_counts, axis=1)
            targets_list.append(_targets)
        curated_target_labels[str(Path(cfg_path).stem)] = targets_list

    # Define class labels for noisy, clean and curated datasets
    df_rows_list = []
    metric_names = ["accuracy", "top_n_accuracy", "cross_entropy", "accuracy_per_class"]

    # Run the same experiment multiple time
    for _run_id in range(number_of_runs):
        target_labels = {"clean": dataset.clean_targets[ind],  # type: ignore
                         "noisy": np.array([dataset.__getitem__(_i)[2] for _i in ind]),
                         **{_n: _l[_run_id] for _n, _l in curated_target_labels.items()}}

        # Loops over different models
        for config_id, config in enumerate([load_model_config(cfg) for cfg in args.config]):
            posteriors = inference_ensemble(dataset, config)[1][ind]

            # Collect metrics
            for _label_name, _label in target_labels.items():
                df_row = {"model": Path(args.config[config_id]).stem, "run_id": _run_id,
                          "dataset": _label_name, "count": _label.size}
                for _metric_name in metric_names:
                    _val = benchmark_metrics(posteriors, observed_labels=_label, metric_name=_metric_name,
                                             true_labels=target_labels["clean"])
                    df_row.update({_metric_name: _val})  # type: ignore
                df_rows_list.append(df_row)

    df = pd.DataFrame(df_rows_list)
    df = df.sort_values(by=["dataset", "model"], axis=0)
    logging.info(f"\n{df.to_string()}")

    # Aggregate multiple runs
    group_cols = ['model', 'dataset']
    df_grouped = df.groupby(group_cols, as_index=False)['accuracy', 'count', 'cross_entropy'].agg([np.mean, np.std])
    logging.info(f"\n{df_grouped.to_string()}")

    # Plot the observed confusion matrix
    plot_confusion_matrix(target_labels["clean"], target_labels["noisy"],
                          get_cifar10_label_names(), save_path=FIGURE_DIR)


def benchmark_metrics(posteriors: np.ndarray,
                      observed_labels: np.ndarray,
                      metric_name: str,
                      true_labels: np.ndarray) -> Union[float, List[float]]:
    """
    Defines metrics to be used in model comparison.
    """
    predictions = np.argmax(posteriors, axis=1)

    # Accuracy averaged across all classes
    if metric_name == "accuracy":
        return np.mean(predictions == observed_labels) * 100.0
    # Cross-entropy loss across all samples
    elif metric_name == "top_n_accuracy":
        N = 2
        sorted_class_predictions = np.argsort(posteriors, axis=1)[:, ::-1]
        correct = int(0)
        for _i in range(observed_labels.size):
            correct += np.any(sorted_class_predictions[_i, :N] == observed_labels[_i])
        return correct * 100.0 / observed_labels.size
    elif metric_name == "cross_entropy":
        return np.mean(cross_entropy(posteriors, np.eye(10)[observed_labels]))
    # Average accuracy per class - samples are groupped based on their true class label
    elif metric_name == "accuracy_per_class":
        vals = list()
        for _class in np.unique(true_labels, return_counts=False):
            mask = true_labels == _class
            val = np.mean(predictions[mask] == observed_labels[mask]) * 100.0
            vals.append(np.around(val, decimals=3))
        return vals
    else:
        raise ValueError("Unknown metric")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Execute benchmark 3')
    parser.add_argument('--config', dest='config', type=str, required=True, nargs='+',
                        help='Path to config file(s) characterising trained CNN model/s')
    parser.add_argument('--curated-label-config', dest='curated_label_config', type=str, required=False, nargs='+',
                        help='Path to config file(s) corresponding to curated labels in adjudication simulation')
    args, unknown_args = parser.parse_known_args()
    create_folder(MODEL_SELECTION_BENCHMARK_DIR)
    create_logger(MODEL_SELECTION_BENCHMARK_DIR)
    main(args)

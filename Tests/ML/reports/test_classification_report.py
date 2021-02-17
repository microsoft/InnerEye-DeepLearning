#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from io import StringIO

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.reports.classification_report import ReportedMetrics, get_correct_and_misclassified_examples, \
    get_image_filepath_from_subject_id, get_k_best_and_worst_performing, get_metric, get_results, \
    plot_image_from_filepath, get_unique_label_combinations, generate_psuedo_labels, get_image_labels_from_subject_id
from InnerEye.ML.reports.notebook_report import generate_classification_notebook
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.configs.classification.DummyMulticlassClassification import DummyMulticlassClassification


def test_generate_classification_report(test_output_dirs: OutputFolderForTests) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"
    dataset_csv_path = reports_folder / 'dataset.csv'

    current_dir = test_output_dirs.make_sub_dir("test_classification_report")
    result_file = current_dir / "report.ipynb"
    config = ScalarModelBase()
    result_html = generate_classification_notebook(result_notebook=result_file,
                                                   config=config,
                                                   val_metrics=val_metrics_file,
                                                   test_metrics=test_metrics_file,
                                                   dataset_csv_path=dataset_csv_path)
    assert result_file.is_file()
    assert result_html.is_file()
    assert result_html.suffix == ".html"


def test_get_results() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    hues = ["Default"]
    results = get_results(test_metrics_file, hues)
    assert all([results.subject_ids[i] == i for i in range(12)])
    assert all([results.labels[i] == label for i, label in enumerate([1] * 6 + [0] * 6)])
    assert all([results.model_outputs[i] == op for i, op in enumerate([0.0, 0.2, 0.4, 0.6, 0.8, 1.0] * 2)])


def test_get_results_multiple_hues(test_output_dirs: OutputFolderForTests) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"

    # Write a new metrics file with 2 hues, label is only associated with one hue
    csv = pd.read_csv(test_metrics_file)
    hues = ["Hue1", "Hue2"]
    csv.loc[::2, LoggingColumns.Hue.value] = hues[0]
    csv.loc[1::2, LoggingColumns.Hue.value] = hues[1]
    csv.loc[::2, LoggingColumns.Patient.value] = list(range(len(csv)//2))
    csv.loc[1::2, LoggingColumns.Patient.value] = list(range(len(csv)//2))
    metrics_csv_multi_hue = test_output_dirs.root_dir / "metrics.csv"
    csv.to_csv(metrics_csv_multi_hue, index=False)

    for h, hue in enumerate(hues):
        results = get_results(metrics_csv_multi_hue, [hue])
        assert all([results.subject_ids[i] == i for i in range(6)])
        assert all([results.labels[i] == label for i, label in enumerate([1] * 3 + [0] * 3)])
        assert all([results.model_outputs[i] == round(op, 2) for i, op in enumerate(list(np.linspace(0.2*h, 0.8 + 0.2*h, 3)) * 2)])


def test_get_unique_label_combinations():
    dataset_csv = full_ml_test_data_path("classification_data_multiclass") / "dataset.csv"
    config = DummyMulticlassClassification()
    unique_labels = get_unique_label_combinations(dataset_csv, config)
    class_names = config.class_names
    assert set(map(tuple, unique_labels)) == set([tuple(class_names[i] for i in labels) for labels in [[1, 2, 3], [2, 3], [3]]])


def test_generate_psuedo_labels():

    csv = StringIO("prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split\n"
                   "Hue1,0,0,0.5,1,-1,Test\n"
                   "Hue2,0,0,0.6,1,-1,Test\n"
                   "Hue3,0,0,0.3,0,-1,Test\n"
                   "Hue1,0,1,0.5,1,-1,Test\n"
                   "Hue2,0,1,0.6,1,-1,Test\n"
                   "Hue3,0,1,0.5,1,-1,Test\n"
                   "Hue1,0,2,0.5,1,-1,Test\n"
                   "Hue2,0,2,0.6,1,-1,Test\n"
                   "Hue3,0,2,0.5,0,-1,Test\n"
                   "Hue1,0,3,0.5,1,-1,Test\n"
                   "Hue2,0,3,0.6,1,-1,Test\n"
                   "Hue3,0,3,0.3,1,-1,Test\n"
                   )

    expected_csv = StringIO("prediction_target,epoch,subject,model_output,label,cross_validation_split_index,data_split\n"
                            "Hue1|Hue2,0,0,1,1,-1,Test\n"
                            "Hue1|Hue2,0,1,0,0,-1,Test\n"
                            "Hue1|Hue2,0,2,0,1,-1,Test\n"
                            "Hue1|Hue2,0,3,1,0,-1,Test\n"
                            )

    df = generate_psuedo_labels(csv=csv,
                                hues=["Hue1", "Hue2"],
                                all_hues=["Hue1", "Hue2", "Hue3"],
                                per_class_thresholds=[0.4, 0.5, 0.4])
    expected_df = pd.read_csv(expected_csv)
    assert expected_df.equals(df)


def test_functions_with_invalid_csv(test_output_dirs: OutputFolderForTests) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"
    invalid_metrics_file = Path(test_output_dirs.root_dir) / "invalid_metrics_classification.csv"
    shutil.copyfile(test_metrics_file, invalid_metrics_file)
    # Duplicate a subject
    with open(invalid_metrics_file, "a") as file:
        file.write("Default,1,5,1.0,1,-1,Test")
    hue = "Default"
    with pytest.raises(ValueError):
        get_results(invalid_metrics_file, [hue])

    with pytest.raises(ValueError):
        get_correct_and_misclassified_examples(invalid_metrics_file, test_metrics_file, hue)

    with pytest.raises(ValueError):
        get_correct_and_misclassified_examples(val_metrics_file, invalid_metrics_file, hue)


def test_get_metric() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    optimal_threshold = get_metric(test_metrics_csv=test_metrics_file,
                                   val_metrics_csv=val_metrics_file,
                                   metric=ReportedMetrics.OptimalThreshold,
                                   hues=["Default"])

    assert optimal_threshold == 0.6

    auc_roc = get_metric(test_metrics_csv=test_metrics_file,
                         val_metrics_csv=val_metrics_file,
                         metric=ReportedMetrics.AUC_ROC,
                         hues=["Default"])
    assert auc_roc == 0.5

    auc_pr = get_metric(test_metrics_csv=test_metrics_file,
                        val_metrics_csv=val_metrics_file,
                        metric=ReportedMetrics.AUC_PR,
                        hues=["Default"])

    assert math.isclose(auc_pr, 13 / 24, abs_tol=1e-15)

    accuracy = get_metric(test_metrics_csv=test_metrics_file,
                          val_metrics_csv=val_metrics_file,
                          metric=ReportedMetrics.Accuracy,
                          hues=["Default"])

    assert accuracy == 0.5

    fpr = get_metric(test_metrics_csv=test_metrics_file,
                     val_metrics_csv=val_metrics_file,
                     metric=ReportedMetrics.FalsePositiveRate,
                     hues=["Default"])

    assert fpr == 0.5

    fnr = get_metric(test_metrics_csv=test_metrics_file,
                     val_metrics_csv=val_metrics_file,
                     metric=ReportedMetrics.FalseNegativeRate,
                     hues=["Default"])

    assert fnr == 0.5


def test_get_correct_and_misclassified_examples() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    results = get_correct_and_misclassified_examples(val_metrics_csv=val_metrics_file,
                                                     test_metrics_csv=test_metrics_file)

    true_positives = [item[LoggingColumns.Patient.value] for _, item in results.true_positives.iterrows()]
    assert all([i in true_positives for i in [3, 4, 5]])

    true_negatives = [item[LoggingColumns.Patient.value] for _, item in results.true_negatives.iterrows()]
    assert all([i in true_negatives for i in [6, 7, 8]])

    false_positives = [item[LoggingColumns.Patient.value] for _, item in results.false_positives.iterrows()]
    assert all([i in false_positives for i in [9, 10, 11]])

    false_negatives = [item[LoggingColumns.Patient.value] for _, item in results.false_negatives.iterrows()]
    assert all([i in false_negatives for i in [0, 1, 2]])


def test_get_k_best_and_worst_performing() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"
    val_metrics_file = reports_folder / "val_metrics_classification.csv"

    results = get_k_best_and_worst_performing(val_metrics_csv=val_metrics_file,
                                              test_metrics_csv=test_metrics_file,
                                              k=2)

    best_true_positives = [item[LoggingColumns.Patient.value] for _, item in results.true_positives.iterrows()]
    assert best_true_positives == [5, 4]

    best_true_negatives = [item[LoggingColumns.Patient.value] for _, item in results.true_negatives.iterrows()]
    assert best_true_negatives == [6, 7]

    worst_false_positives = [item[LoggingColumns.Patient.value] for _, item in results.false_positives.iterrows()]
    assert worst_false_positives == [11, 10]

    worst_false_negatives = [item[LoggingColumns.Patient.value] for _, item in results.false_negatives.iterrows()]
    assert worst_false_negatives == [0, 1]


def test_get_image_filepath_from_subject_id_single() -> None:
    reports_folder = Path(__file__).parent
    dataset_csv_file = reports_folder / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_file)

    config = DummyMulticlassClassification()
    config.subject_column = "subject"
    config.image_file_column = "filePath"

    filepath = get_image_filepath_from_subject_id(subject_id="1",
                                                  dataset_df=dataset_df,
                                                  config=config,
                                                  dataset_dir=reports_folder)
    expected_path = Path(reports_folder / "../test_data/classification_data_2d/im2.npy")

    assert filepath
    assert len(filepath) == 1
    assert expected_path.samefile(filepath[0])


def test_get_image_filepath_from_subject_id_multiple() -> None:
    reports_folder = Path(__file__).parent
    dataset_csv_file = reports_folder / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_file)
    config = DummyMulticlassClassification()
    config.subject_column = "subject"
    config.image_file_column = "filePath"

    # duplicate subject entries
    dataset_df.loc[::2, config.subject_column] = list(range(len(dataset_df)//2))
    dataset_df.loc[1::2, config.subject_column] = list(range(len(dataset_df)//2))

    filepath = get_image_filepath_from_subject_id(subject_id="1",
                                                  dataset_df=dataset_df,
                                                  config=config,
                                                  dataset_dir=reports_folder)
    expected_paths = [Path(reports_folder / "../test_data/classification_data_2d/im1.npy"),
                      Path(reports_folder / "../test_data/classification_data_2d/im2.npy")]

    assert filepath
    assert len(filepath) == 2
    assert expected_paths[0].samefile(filepath[0])
    assert expected_paths[1].samefile(filepath[1])


def test_get_image_filepath_from_subject_id_invalid_id() -> None:
    reports_folder = Path(__file__).parent
    dataset_csv_file = reports_folder / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_file)
    config = DummyMulticlassClassification()
    config.subject_column = "subject"
    config.image_file_column = "filePath"

    filepath = get_image_filepath_from_subject_id(subject_id="100",
                                                  dataset_df=dataset_df,
                                                  config=config,
                                                  dataset_dir=reports_folder)

    assert not filepath


def test_image_labels_from_subject_id_single() -> None:
    reports_folder = Path(__file__).parent
    dataset_csv_file = reports_folder / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_file)

    config = DummyMulticlassClassification()
    config.subject_column = "subject"

    labels = get_image_labels_from_subject_id(subject_id="1",
                                                  dataset_df=dataset_df,
                                                  config=config)
    assert labels
    assert len(labels) == 1
    assert labels[0] == "class0"


def test_image_labels_from_subject_id_multiple() -> None:
    reports_folder = Path(__file__).parent
    dataset_csv_file = reports_folder / "dataset.csv"
    dataset_df = pd.read_csv(dataset_csv_file)

    config = DummyMulticlassClassification()
    config.subject_column = "subject"

    # Add multiple labels for a single subject
    dataset_df.loc[:, config.label_value_column] = "1|2"

    labels = get_image_labels_from_subject_id(subject_id="1",
                                                  dataset_df=dataset_df,
                                                  config=config)
    assert labels
    assert len(labels) == 2
    assert set(labels) == {"class1", "class2"}


def test_plot_image_from_filepath(test_output_dirs: OutputFolderForTests) -> None:
    im_width = 200

    array = np.ones([10, 10])
    valid_file = Path(test_output_dirs.root_dir) / "valid.npy"
    np.save(valid_file, array)
    res = plot_image_from_filepath(valid_file, im_width)
    assert res

    array = np.ones([3, 10, 10])
    invalid_file = Path(test_output_dirs.root_dir) / "invalid.npy"
    np.save(invalid_file, array)
    res = plot_image_from_filepath(invalid_file, im_width)
    assert not res

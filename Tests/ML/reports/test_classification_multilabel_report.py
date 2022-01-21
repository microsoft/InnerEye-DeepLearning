#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.configs.classification.DummyMulticlassClassification import DummyMulticlassClassification
from InnerEye.ML.metrics_dict import MetricsDict
from InnerEye.ML.reports.classification_multilabel_report import get_dataframe_with_exact_label_matches, \
    get_labels_and_predictions_for_prediction_target_set, get_unique_prediction_target_combinations
from InnerEye.ML.reports.notebook_report import generate_classification_multilabel_notebook
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.Azure.azure_util import DEFAULT_CROSS_VALIDATION_SPLIT_INDEX


def test_generate_classification_multilabel_report(test_output_dirs: OutputFolderForTests) -> None:
    hues = ["Hue1", "Hue2"]

    config = ScalarModelBase(label_value_column="label",
                             image_file_column="filePath",
                             image_channels=["image1", "image2"],
                             label_channels=["image1"])
    config.class_names = hues

    test_metrics_file = test_output_dirs.root_dir / "test_metrics_classification.csv"
    val_metrics_file = test_output_dirs.root_dir / "val_metrics_classification.csv"

    config.local_dataset = test_output_dirs.root_dir / "dataset"
    config.local_dataset.mkdir()
    dataset_csv_path = config.local_dataset / "dataset.csv"
    image_file_name = "image.npy"

    pd.DataFrame.from_dict({LoggingColumns.Hue.value: [hues[0], hues[1]] * 6,
                            LoggingColumns.Epoch.value: [0] * 12,
                            LoggingColumns.Patient.value: [s for s in range(6) for _ in range(2)],
                            LoggingColumns.ModelOutput.value: [0.1, 0.1, 0.1, 0.9, 0.1, 0.9,
                                                               0.9, 0.9, 0.9, 0.9, 0.9, 0.1],
                            LoggingColumns.Label.value: [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
                            LoggingColumns.CrossValidationSplitIndex: [DEFAULT_CROSS_VALIDATION_SPLIT_INDEX] * 12,
                            LoggingColumns.DataSplit.value: [0] * 12,
                            }).to_csv(test_metrics_file, index=False)

    pd.DataFrame.from_dict({LoggingColumns.Hue.value: [hues[0], hues[1]] * 6,
                            LoggingColumns.Epoch.value: [0] * 12,
                            LoggingColumns.Patient.value: [s for s in range(6) for _ in range(2)],
                            LoggingColumns.ModelOutput.value: [0.1, 0.1, 0.1, 0.1, 0.1, 0.9,
                                                               0.9, 0.9, 0.9, 0.1, 0.9, 0.1],
                            LoggingColumns.Label.value: [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0],
                            LoggingColumns.CrossValidationSplitIndex: [DEFAULT_CROSS_VALIDATION_SPLIT_INDEX] * 12,
                            LoggingColumns.DataSplit.value: [0] * 12,
                            }).to_csv(val_metrics_file, index=False)

    pd.DataFrame.from_dict({config.subject_column: [s for s in range(6) for _ in range(2)],
                            config.channel_column: ["image1", "image2"] * 6,
                            config.image_file_column: [f for f in [f"0_{image_file_name}", f"1_{image_file_name}"]
                                                       for _ in range(6)],
                            config.label_value_column: ["", "", "1", "1", "1", "1", "0|1", "0|1", "0|1", "0|1", "0",
                                                        "0"]
                            }).to_csv(dataset_csv_path, index=False)

    np.save(str(Path(config.local_dataset / f"0_{image_file_name}")),
            np.random.randint(0, 255, [5, 4]))
    np.save(str(Path(config.local_dataset / f"1_{image_file_name}")),
            np.random.randint(0, 255, [5, 4]))

    result_file = test_output_dirs.root_dir / "report.ipynb"
    result_html = generate_classification_multilabel_notebook(result_notebook=result_file,
                                                              config=config,
                                                              val_metrics=val_metrics_file,
                                                              test_metrics=test_metrics_file)
    assert result_file.is_file()
    assert result_html.is_file()
    assert result_html.suffix == ".html"


def test_get_pseudo_labels_and_predictions() -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"

    results = get_labels_and_predictions_for_prediction_target_set(test_metrics_file,
                                                                   [MetricsDict.DEFAULT_HUE_KEY],
                                                                   all_prediction_targets=[MetricsDict.DEFAULT_HUE_KEY],
                                                                   thresholds_per_prediction_target=[0.5])
    assert all([results.subject_ids[i] == i for i in range(12)])
    assert all([results.labels[i] == label for i, label in enumerate([1] * 6 + [0] * 6)])
    assert all([results.model_outputs[i] == op for i, op in enumerate([0.0, 0.0, 0.0, 1.0, 1.0, 1.0] * 2)])


def test_get_pseudo_labels_and_predictions_multiple_hues(test_output_dirs: OutputFolderForTests) -> None:
    reports_folder = Path(__file__).parent
    test_metrics_file = reports_folder / "test_metrics_classification.csv"

    # Write a new metrics file with 2 prediction targets,
    # prediction_target_set_to_match will only be associated with one prediction target
    csv = pd.read_csv(test_metrics_file)
    hues = ["Hue1", "Hue2"]
    csv.loc[::2, LoggingColumns.Hue.value] = hues[0]
    csv.loc[1::2, LoggingColumns.Hue.value] = hues[1]
    csv.loc[::2, LoggingColumns.Patient.value] = list(range(len(csv) // 2))
    csv.loc[1::2, LoggingColumns.Patient.value] = list(range(len(csv) // 2))
    csv.loc[::2, LoggingColumns.Label.value] = [0, 0, 0, 1, 1, 1]
    csv.loc[1::2, LoggingColumns.Label.value] = [0, 1, 1, 1, 1, 0]
    csv.loc[::2, LoggingColumns.ModelOutput.value] = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
    csv.loc[1::2, LoggingColumns.ModelOutput.value] = [0.1, 0.9, 0.9, 0.9, 0.9, 0.1]
    metrics_csv_multi_hue = test_output_dirs.root_dir / "metrics.csv"
    csv.to_csv(metrics_csv_multi_hue, index=False)

    for h, hue in enumerate(hues):
        results = get_labels_and_predictions_for_prediction_target_set(metrics_csv_multi_hue,
                                                                       prediction_target_set_to_match=[hue],
                                                                       all_prediction_targets=hues,
                                                                       thresholds_per_prediction_target=[0.5, 0.5])
        assert all([results.subject_ids[i] == i for i in range(6)])
        assert all([results.labels[i] == label
                    for i, label in enumerate([0, 0, 0, 0, 0, 1] if h == 0 else [0, 1, 1, 0, 0, 0])])
        assert all([results.model_outputs[i] == op
                    for i, op in enumerate([0, 0, 0, 0, 0, 1] if h == 0 else [0, 1, 1, 0, 0, 0])])


def test_generate_pseudo_labels() -> None:
    metrics_df = pd.DataFrame.from_dict({"prediction_target": ["Hue1", "Hue2", "Hue3"] * 4,
                                         "epoch": [0] * 12,
                                         "subject": [i for i in range(4) for _ in range(3)],
                                         "model_output": [0.5, 0.6, 0.3, 0.5, 0.6, 0.5, 0.5, 0.6, 0.5, 0.5, 0.6, 0.3],
                                         "label": [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1],
                                         "cross_validation_split_index": [DEFAULT_CROSS_VALIDATION_SPLIT_INDEX] * 12,
                                         "data_split": [ModelExecutionMode.TEST.value] * 12
                                         })

    expected_df = pd.DataFrame.from_dict({"subject": list(range(4)),
                                          "model_output": [1, 0, 0, 1],
                                          "label": [1, 0, 1, 0],
                                          "prediction_target": ["Hue1|Hue2"] * 4})

    df = get_dataframe_with_exact_label_matches(metrics_df=metrics_df,
                                                prediction_target_set_to_match=["Hue1", "Hue2"],
                                                all_prediction_targets=["Hue1", "Hue2", "Hue3"],
                                                thresholds_per_prediction_target=[0.4, 0.5, 0.4])
    assert expected_df.equals(df)


def test_generate_pseudo_labels_negative_class() -> None:
    metrics_df = pd.DataFrame.from_dict({"prediction_target": ["Hue1", "Hue2", "Hue3"] * 4,
                                         "epoch": [0] * 12,
                                         "subject": [i for i in range(4) for _ in range(3)],
                                         "model_output": [0.2, 0.3, 0.2, 0.5, 0.6, 0.5, 0.5, 0.6, 0.5, 0.2, 0.3, 0.2],
                                         "label": [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1],
                                         "cross_validation_split_index": [DEFAULT_CROSS_VALIDATION_SPLIT_INDEX] * 12,
                                         "data_split": [ModelExecutionMode.TEST.value] * 12
                                         })

    expected_df = pd.DataFrame.from_dict({"subject": list(range(4)),
                                          "model_output": [1, 0, 0, 1],
                                          "label": [1, 1, 0, 0],
                                          "prediction_target": [""] * 4})

    df = get_dataframe_with_exact_label_matches(metrics_df=metrics_df,
                                                prediction_target_set_to_match=[],
                                                all_prediction_targets=["Hue1", "Hue2", "Hue3"],
                                                thresholds_per_prediction_target=[0.4, 0.5, 0.4])
    assert expected_df.equals(df)


def test_get_unique_label_combinations_single_label(test_output_dirs: OutputFolderForTests) -> None:
    config = ScalarModelBase(label_channels=["label"],
                             label_value_column="value",
                             image_channels=["image"],
                             image_file_column="path",
                             subject_column="subjectID")
    class_names = config.class_names

    config.local_dataset = test_output_dirs.root_dir / "dataset"
    config.local_dataset.mkdir()
    dataset_csv = config.local_dataset / "dataset.csv"
    dataset_csv.write_text("subjectID,channel,path,value\n"
                           "S1,label,random,1\n"
                           "S1,image,random,\n"
                           "S2,label,random,0\n"
                           "S2,image,random,\n"
                           "S3,label,random,1\n"
                           "S3,image,random,\n")

    unique_labels = get_unique_prediction_target_combinations(config)  # type: ignore
    expected_label_combinations = set(frozenset(class_names[i] for i in labels)  # type: ignore
                                      for labels in [[], [0]])
    assert unique_labels == expected_label_combinations


def test_get_unique_label_combinations_nan(test_output_dirs: OutputFolderForTests) -> None:
    config = ScalarModelBase(label_channels=["label"],
                             label_value_column="value",
                             image_channels=["image"],
                             image_file_column="path",
                             subject_column="subjectID")
    class_names = config.class_names

    config.local_dataset = test_output_dirs.root_dir / "dataset"
    config.local_dataset.mkdir()
    dataset_csv = config.local_dataset / "dataset.csv"
    dataset_csv.write_text("subjectID,channel,path,value\n"
                           "S1,label,random,1\n"
                           "S1,image,random,\n"
                           "S2,label,random,\n"
                           "S2,image,random,\n")
    unique_labels = get_unique_prediction_target_combinations(config)  # type: ignore
    expected_label_combinations = set(frozenset(class_names[i] for i in labels)  # type: ignore
                                      for labels in [[0]])
    assert unique_labels == expected_label_combinations


def test_get_unique_label_combinations_multi_label(test_output_dirs: OutputFolderForTests) -> None:
    config = DummyMulticlassClassification()
    class_names = config.class_names
    config.local_dataset = test_output_dirs.root_dir / "dataset"
    config.local_dataset.mkdir()
    dataset_csv = config.local_dataset / "dataset.csv"
    dataset_csv.write_text("ID,channel,path,label\n"
                           "S1,blue,random,1|2|3\n"
                           "S1,green,random,\n"
                           "S2,blue,random,2|3\n"
                           "S2,green,random,\n"
                           "S3,blue,random,3\n"
                           "S3,green,random,\n"
                           "S4,blue,random,\n"
                           "S4,green,random,\n")
    unique_labels = get_unique_prediction_target_combinations(config)  # type: ignore

    expected_label_combinations = set(frozenset(class_names[i] for i in labels)  # type: ignore
                                      for labels in [[1, 2, 3], [2, 3], [3], []])
    assert unique_labels == expected_label_combinations

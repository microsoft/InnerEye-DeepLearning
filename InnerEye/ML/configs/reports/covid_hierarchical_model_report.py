import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.reports.classification_report import get_labels_and_predictions_from_dataframe, LabelsAndPredictions
from InnerEye.ML.reports.notebook_report import print_table
from InnerEye.ML.scalar_config import ScalarModelBase

TARGET_NAMES = ['CVX03vs12', 'CVX0vs3', 'CVX1vs2']
MULTICLASS_HUE_NAME = "Multiclass"


def get_label_from_label_dict(label_dict: Dict[str, float]) -> int:
    """
    Converts strings CVX03vs12, CVX1vs2, CVX0vs3 to the corresponding class as int.
    """
    if label_dict['CVX03vs12'] == 0:
        assert np.isnan(label_dict['CVX1vs2'])
        if label_dict['CVX0vs3'] == 0:
            label = 0
        elif label_dict['CVX0vs3'] == 1:
            label = 3
        else:
            raise ValueError("CVX0vs3 should be 0 or 1.")
    elif label_dict['CVX03vs12'] == 1:
        assert np.isnan(label_dict['CVX0vs3'])
        if label_dict['CVX1vs2'] == 0:
            label = 1
        elif label_dict['CVX1vs2'] == 1:
            label = 2
        else:
            raise ValueError("CVX1vs2 should be 0 or 1.")
    else:
        raise ValueError("CVX03vs12 should be 0 or 1.")
    return label


def get_model_prediction_by_probabilities(output_dict: Dict[str, float]) -> int:
    """
    Based on the values for CVX03vs12, CVX0vs3 and CVX1vs2 predicted by the model, predict the CVX scores as followed:
    score(CVX0) = [1 - score(CVX03vs12)][1 - score(CVX0vs3)]
    score(CVX1) = score(CVX03vs12)[1 - score(CVX1vs2)]
    score(CVX2) = score(CVX03vs12)score(CVX1vs2)
    score(CVX3) = [1 - score(CVX03vs12)]score(CVX0vs3)
    """
    cvx0 = (1 - output_dict['CVX03vs12']) * (1 - output_dict['CVX0vs3'])
    cvx3 = (1 - output_dict['CVX03vs12']) * output_dict['CVX0vs3']
    cvx1 = output_dict['CVX03vs12'] * (1 - output_dict['CVX1vs2'])
    cvx2 = output_dict['CVX03vs12'] * output_dict['CVX1vs2']
    return np.argmax([cvx0, cvx1, cvx2, cvx3])


def get_dataframe_with_covid_labels(metrics_df: pd.DataFrame) -> pd.DataFrame:
    def get_CVX_labels(df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe (with only one subject) with the model outputs for CVX03vs12, CVX0vs3 and CVX1vs2,
        returns a corresponding dataframe with scores for CVX0, CVX1, CVX2 and CVX3 for this subject. See
        `get_model_prediction_by_probabilities` for details on mapping the model output to CVX labels.
        """
        df_by_hue = df[df[LoggingColumns.Hue.value].isin(TARGET_NAMES)].set_index(LoggingColumns.Hue.value)
        model_output = get_model_prediction_by_probabilities(df_by_hue[LoggingColumns.ModelOutput.value].to_dict())
        label = get_label_from_label_dict(df_by_hue[LoggingColumns.Label.value].to_dict())

        return pd.DataFrame.from_dict({LoggingColumns.Patient.value: [df.iloc[0][LoggingColumns.Patient.value]],
                                       LoggingColumns.ModelOutput.value: [model_output],
                                       LoggingColumns.Label.value: [label]})

    df = metrics_df.copy()
    # Group by subject, and for each subject, convert the CVX03vs12, CVX0vs3 and CVX1vs2 predictions to CVX labels.
    df = df.groupby(LoggingColumns.Patient.value, as_index=False).apply(get_CVX_labels).reset_index(drop=True)
    df[LoggingColumns.Hue.value] = [MULTICLASS_HUE_NAME] * len(df)
    return df


def get_labels_and_predictions_covid_labels(csv: Path) -> LabelsAndPredictions:
    metrics_df = pd.read_csv(csv)
    df = get_dataframe_with_covid_labels(metrics_df=metrics_df)
    return get_labels_and_predictions_from_dataframe(df)


def print_metrics_from_csv(csv_to_set_optimal_threshold: Path,
                           csv_to_compute_metrics: Path,
                           config: ScalarModelBase,
                           is_crossval_report: bool) -> None:
    assert config.target_names == TARGET_NAMES

    predictions_to_compute_metrics = get_labels_and_predictions_covid_labels(
        csv=csv_to_compute_metrics)

    acc = accuracy_score(predictions_to_compute_metrics.labels, predictions_to_compute_metrics.model_outputs)
    rows = [[f"{acc:.4f}"]]
    print_table(rows, header=["Multiclass Accuracy"])

    conf_matrix = confusion_matrix(predictions_to_compute_metrics.labels, predictions_to_compute_metrics.model_outputs)
    rows = []
    header = ["", "CVX0 predicted", "CVX1 predicted", "CVX2 predicted", "CVX3 predicted"]
    for i in range(conf_matrix.shape[0]):
        line = [f"CVX{i} GT"] + list(conf_matrix[i])
        rows.append(line)
    print_table(rows, header=header)

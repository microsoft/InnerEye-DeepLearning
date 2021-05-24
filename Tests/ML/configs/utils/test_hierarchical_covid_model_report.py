import pandas as pd
from math import nan

from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.configs.reports.covid_hierarchical_model_report import MULTICLASS_HUE_NAME, \
    get_dataframe_with_covid_labels


def test_get_dataframe_with_covid_labels() -> None:

    df = pd.DataFrame.from_dict({LoggingColumns.Patient.value: [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                                 LoggingColumns.Hue.value: ['CVX03vs12', 'CVX0vs3', 'CVX1vs2'] * 4,
                                 LoggingColumns.Label.value: [0, 0, nan, 0, 1, nan, 1, nan, 0, 1, nan, 1],
                                 LoggingColumns.ModelOutput.value: [0.1, 0.1, 0.5, 0.1, 0.9, 0.5, 0.9, 0.9, 0.9, 0.1, 0.2, 0.1]})
    expected_df = pd.DataFrame.from_dict({LoggingColumns.Patient.value: [1, 2, 3, 4],
                                          LoggingColumns.ModelOutput.value: [0, 3, 2, 0],
                                          LoggingColumns.Label.value: [0, 3, 1, 2],
                                          LoggingColumns.Hue.value: [MULTICLASS_HUE_NAME] * 4
                                          })

    multiclass_df = get_dataframe_with_covid_labels(df)
    assert expected_df.equals(multiclass_df)

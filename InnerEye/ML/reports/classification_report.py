#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve

from InnerEye.ML.reports.notebook_report import print_header


def plot_auc(x_values, y_values, title, ax):
    ax.plot(x_values, y_values)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_title(title)

    # write values of points
    for x, y in zip (x_values, y_values):
        ax.annotate(f"{x:0.3f}, {y:0.3f}", xy = (x, y), xytext=(15, 0), textcoords='offset points')


def plot_pr_and_roc_curves_from_csv(metrics_csv):
    print_header("ROC and PR curves", level=3)
    df = pd.read_csv(metrics_csv)

    _, ax = plt.subplots(1, 2)

    fpr, tpr, thresholds = roc_curve(df["label"], df["model_output"])
    plot_auc(fpr, tpr, "ROC Curve", ax[0])
    precision, recall, thresholds = precision_recall_curve(df["label"], df["model_output"])
    plot_auc(precision, recall, "PR Curve", ax[1])

    plt.show()

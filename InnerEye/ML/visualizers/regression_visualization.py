#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_variation_error_prediction(
        labels: np.ndarray,
        predictions: np.ndarray,
        filename: Optional[str] = None) -> None:
    """
    Plots the absolute prediction errors as well as the predicted values
    against the ground truth values.
    :param labels: ground truth labels
    :param predictions: model outputs
    :param filename: location to save the plot to. If None show the plot instead.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    errors = np.abs(predictions-labels)
    ax[0].scatter(labels, errors, marker="x")
    ax[0].set_xlabel("Ground truth")
    ax[0].set_ylabel("Absolute error")
    ax[0].set_title("Error in function of ground truth value")

    ax[1].scatter(labels, predictions, marker="x")
    # noinspection PyArgumentList
    x = np.linspace(labels.min(), labels.max(), 10)
    ax[1].plot(x, x, "--", linewidth=0.5)
    ax[1].set_xlabel("Ground truth")
    ax[1].set_ylabel("Predicted value")
    ax[1].set_title("Predicted value in function of ground truth")

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, dpi=75)

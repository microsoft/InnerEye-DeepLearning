#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def plot_reliability_curve(
        y_predict: Union[List[np.ndarray], np.ndarray],
        y_true: Union[List[np.ndarray], np.ndarray],
        num_bins: int = 15,
        normalise: bool = False) -> None:
    """
    Plots reliability curves for multiple models to observe model calibration errors.
    Inputs can be either 1-D or a list of 1-D arrays depending on the use case.
    List elements are intended to be used for different model types, e.g. y_predict: (num_samples, num_models)
    :param y_predict: Model predictions, either a 1D array (num_samples) or list of 1D arrays (num_samples, num_models)
    :param y_true: Target values {0, 1}  either a 1D array (num_samples) or list of 1D arrays (num_samples, num_models)
                   Assuming a binary classification case
    :param num_bins: Number of bins used for model prediction probabilities.
    :param normalise: If set to true, predictions are normalised to range [0, 1]

    References
    [1] Predicting Good Probabilities with Supervised Learning
    <https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf>
    """

    if not isinstance(y_predict, list):
        y_predict = [y_predict]
        y_true = [y_true]
    if not len(y_true) == len(y_predict):
        raise ValueError("y_true and y_predict are not of the same length")

    # Generate the figure and axes
    plt.figure(0, figsize=(6, 6))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Iterate over all models and plot y_prediction values
    for model_id in range(len(y_predict)):
        _p = y_predict[model_id]
        _l = y_true[model_id]

        # Remove nan elements from both sets
        mask = ~np.isnan(_l)
        _p = _p[mask]
        _l = _l[mask]

        if _p.shape != _l.shape:
            raise ValueError("Target label and predictions are not of same shape")

        if normalise:
            _p = (_p - _p.min()) / (_p.max() - _p.min())

        # noinspection PyArgumentList
        clf_score = brier_score_loss(_l, _p, pos_label=_l.max())
        frac_of_positives, mean_predicted_value = calibration_curve(_l, _p, n_bins=num_bins)
        ax1.plot(mean_predicted_value, frac_of_positives, "s-", label="%s (%1.3f)" % (f"Model_{model_id}", clf_score))
        ax2.hist(_p, range=(0, 1), bins=num_bins, label=f"Model_{model_id}", histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)')
    ax1.grid()

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)
    ax2.grid()

    plt.tight_layout()
    plt.show()

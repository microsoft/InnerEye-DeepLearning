#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np

from InnerEye.ML.visualizers.reliability_curve import plot_reliability_curve


def test_plot_reliability_curve() -> None:
    prediction = [np.random.rand(250, 1), np.random.rand(200, 1)]
    target = [np.random.randint(2, size=(250, 1)), np.random.randint(2, size=(200, 1))]
    plot_reliability_curve(y_predict=prediction, y_true=target, num_bins=10, normalise=True)

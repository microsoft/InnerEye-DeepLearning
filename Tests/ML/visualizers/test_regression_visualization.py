#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os

import numpy as np

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.visualizers.regression_visualization import plot_variation_error_prediction


def test_plot_variation_errors_for_regression(test_output_dirs: OutputFolderForTests) -> None:
    plot_variation_error_prediction(
        labels=np.array([10, 20, 20, 40, 10, 60, 90]),
        predictions=np.array([12, 25, 10, 36, 11, 69, 90]),
        filename=os.path.join(test_output_dirs.root_dir, "error_plot.png"))
    assert os.path.isfile(os.path.join(test_output_dirs.root_dir, "error_plot.png"))

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.configs.other.DummyEnsembleRegressionContainer import (DummyEnsembleRegressionContainer,
    DummyEnsembleRegressionModule)

def test_local_cross_validation_training_then_inference(test_output_dirs: OutputFolderForTests) -> None:
    """
    Temporary test so I can work through the implementation of cross validation in DummyEnsembleRegressionModule
    """
    container = DummyEnsembleRegressionContainer()
    container.outputs_folder = test_output_dirs
    model = container.model
    model.train()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# import pytest
from pathlib import Path

from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME
from Tests.Common.test_util import DEFAULT_RUN_RECOVERY_ID_NUMERIC
from InnerEye.Scripts.submit_for_inference import main


def test_submit_for_inference():
    args = ["--image_file", "Tests/ML/test_data/train_and_test_data/id1_channel1.nii.gz",
            "--model_id", DEFAULT_RUN_RECOVERY_ID_NUMERIC,
            "--yaml_file", "InnerEye/train_variables.yml",
            "--download_folder", "."]
    main(args)
    assert Path(DEFAULT_RESULT_IMAGE_NAME).exists()

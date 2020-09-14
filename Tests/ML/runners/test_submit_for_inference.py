#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import pytest
from pathlib import Path

from InnerEye.Common import common_util
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME
from Tests.Common.test_util import DEFAULT_MODEL_ID_NUMERIC
from InnerEye.Scripts.submit_for_inference import main


@pytest.mark.skipif(common_util.is_windows(), reason="Testing on Linux is enough")
# Test does not actually need GPU, but we put it in the GPU set as it takes a while to run
# and we don't want to make the main set even longer.
@pytest.mark.gpu
def test_submit_for_inference() -> None:
    args = ["--image_file", "Tests/ML/test_data/train_and_test_data/id1_channel1.nii.gz",
            "--model_id", DEFAULT_MODEL_ID_NUMERIC,
            "--yaml_file", "InnerEye/train_variables.yml",
            "--download_folder", "."]
    seg_path = Path(DEFAULT_RESULT_IMAGE_NAME)
    if seg_path.exists():
        seg_path.unlink()
    main(args)
    assert seg_path.exists()

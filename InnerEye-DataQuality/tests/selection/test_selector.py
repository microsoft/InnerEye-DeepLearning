#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from InnerEyeDataQuality.selection.selectors.base import SampleSelector
import numpy as np


def test_validate_annotation_request() -> None:
    sample_selector = SampleSelector(num_classes=3, num_samples=3)

    # Correct request
    sample_selector.validate_annotation_request(max_cases=2)
    with pytest.raises(RuntimeError):
        sample_selector.validate_annotation_request(max_cases=4)

    # Simulate annotation
    sample_selector.record_selected_cases(np.asarray([0, 1]))

    # Now there are only 1 cases left to annotate
    with pytest.raises(RuntimeError):
        sample_selector.validate_annotation_request(max_cases=2)
    sample_selector.validate_annotation_request(max_cases=1)

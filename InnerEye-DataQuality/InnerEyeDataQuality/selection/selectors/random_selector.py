#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
from InnerEyeDataQuality.selection.selectors.label_based import SampleSelector

class RandomSelector(SampleSelector):
    """
    Selects samples at random
    """
    def __init__(self, num_samples: int, num_classes: int, name: str = "Random Selector") -> None:
        super().__init__(num_samples=num_samples, num_classes=num_classes, name=name)

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return np.random.choice(self.num_samples, self.num_samples, replace=False)

    def get_ambiguity_scores(self, current_labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_mislabelled_scores(self, current_labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
from typing import List, Generator, Optional, Any, Dict
import numpy as np


class SampleSelector(object):
    """
    Sample selector hold a set of label distribution and identifies the next sample to be annotated.
    """

    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 name: str = "Base Selector",
                 allow_repeat_samples: bool = False,
                 use_active_relabelling: bool = False,
                 embeddings: Optional[np.ndarray] = None,
                 output_directory: Optional[Path] = None) -> None:

        super().__init__()
        """
        :param num_samples: Number of samples in the dataset
        :param num_classes: Number of classes of the dataset
        :param initial_labels: The initial labels
        :param name: The name given to the selector
        :param allow_repeat_samples: Whether to allow the sampler to choose the same sample more than once
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.name = name
        self._already_selected_cases: List[int] = list()
        self._allow_repeat_samples = allow_repeat_samples
        self.use_active_relabelling = use_active_relabelling
        self.embeddings = embeddings
        self.output_directory = output_directory
        # Used to store sample scoring and correctness stats for debugging purposes.
        self.stats: Dict[str, Any] = dict()

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        """
        Compute a relabelling score for each sample. The higher the score the better it is to relabel the sample
        :param current_labels: The current labels for each sample.
        :return: An array with a relabelling score for every sample
        """
        raise NotImplementedError

    def get_ambiguity_scores(self, current_labels: np.ndarray) -> np.ndarray:
        """
        Compute an ambiguity score for each sample (higher is more ambiguous).
        :param current_labels: The current labels for each sample.
        :return: An array with an ambiguity score for each sample.
        """
        raise NotImplementedError

    def get_mislabelled_scores(self, current_labels: np.ndarray) -> np.ndarray:
        """
        Compute a mislabelled score for each sample (higher is more likely of being mislabelled).
        :param current_labels: The current labels for each sample.
        :return: An array with a mislabelled score for each sample.
        """
        raise NotImplementedError

    def get_batch_of_samples_to_annotate(self, current_labels: np.ndarray, max_cases: int = 1) -> np.ndarray:
        """

        :param current_labels: The current label counts for each sample shape=(num_samples, num_classes)
        :param max_cases: The number of distinct sample ids to return
        :return:
        """
        self.validate_annotation_request(max_cases)
        sample_scores = self.get_relabelling_scores(current_labels)
        sample_ids = np.argsort(sample_scores)[::-1]
        if not self._allow_repeat_samples:
            mask = np.ones(self.num_samples, dtype=np.bool)
            mask[self._already_selected_cases] = False
            sample_ids = sample_ids[mask[sample_ids]]
        sample_ids = sample_ids[:max_cases]
        self.record_selected_cases(sample_ids)

        return sample_ids

    def record_selected_cases(self, sample_ids: np.ndarray) -> None:
        self._already_selected_cases += sample_ids.tolist()

    def get_available_case_ids(self) -> Generator:
        for i in range(self.num_samples):
            if i not in self._already_selected_cases:
                yield i

    def validate_annotation_request(self, max_cases: int) -> None:
        n_cases_left = self.num_samples - len(self._already_selected_cases)
        if max_cases > n_cases_left:
            raise RuntimeError(
                f"Relabeling was requested for {max_cases} cases but there are only {n_cases_left} to annotate.")

    def update_beliefs(self, iteration_id: int, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

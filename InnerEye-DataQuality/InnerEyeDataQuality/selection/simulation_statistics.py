#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, List, Dict, Any
import numpy as np
from InnerEyeDataQuality.evaluation.metrics import compute_accuracy, compute_label_entropy, total_variation

STAT_FIELDS = ["relabelling_score", "ambiguity", "label_correctness"]

@dataclass(frozen=True)
class SelectionType(Enum):
    """
    Defines the 5 possible types of selections that can be made in an iteration
    """
    MISLABELLED_CASE_SELECTED_CORRECTED = 1
    MISLABELLED_CASE_SELECTED_NOT_CORRECTED = 2
    AMBIGUOUS_CASE_SELECTED_CORRECTED = 3
    AMBIGUOUS_CASE_SELECTED_NOT_CORRECTED = 4
    CLEAN_CASE_SELECTED = 5

def compute_selection_type_of_current_iter(sample_id: int,
                                           true_ambiguous_cases: np.ndarray,
                                           true_label_counts: np.ndarray,
                                           mislabelled_ids_current: np.ndarray,
                                           ambiguous_case_ids_current: np.ndarray,
                                           mislabelled_ids_prev: np.ndarray,
                                           ambiguous_case_ids_prev: np.ndarray) -> SelectionType:
    """
    Compute the type of selection that occurred between the previous and current iteration.
    :param sample_id: The sample id.
    :param true_ambiguous_cases: The ids for the true ambiguous samples.
    :param true_label_counts: The label counts for the true label distribution.
    :param mislabelled_ids_current: The ids for the current iteration remaining not ambiguous mislabelled samples.
    :param ambiguous_case_ids_current: The ids for the current iteration remaining ambiguous mislabelled samples.
    :param mislabelled_ids_prev: The ids for the previous iteration remaining not ambiguous mislabelled samples.
    :param ambiguous_case_ids_prev: The ids for the previous iteration remaining ambiguous mislabelled samples.
    :return: An enum representing the selection type that occurred between the previous and current iteration.
    """
    if sample_id in true_ambiguous_cases:
        if len(set(ambiguous_case_ids_prev) - set(ambiguous_case_ids_current)) > 0:
            return SelectionType.AMBIGUOUS_CASE_SELECTED_CORRECTED
        else:
            return SelectionType.AMBIGUOUS_CASE_SELECTED_NOT_CORRECTED
    else:
        if len(set(mislabelled_ids_prev) - set(mislabelled_ids_current)) > 0:
            return SelectionType.MISLABELLED_CASE_SELECTED_CORRECTED
        elif len(np.unique(np.where(true_label_counts[sample_id])[0])) == 1:
            return SelectionType.CLEAN_CASE_SELECTED
        else:
            return SelectionType.MISLABELLED_CASE_SELECTED_NOT_CORRECTED

def get_mislabelled_sample_ids(true_label_counts: np.ndarray, current_label_counts: np.ndarray) -> np.ndarray:
    """
    Compute which samples are mislabelled.
    :param true_label_counts: The label counts for the true label distribution.
    :param current_label_counts: The label counts for the current distribution.
    :return: An array with the ids of the mislabeled samples (majority voting)
    """
    true_class = np.argmax(true_label_counts, axis=1)
    current_class = np.argmax(current_label_counts, axis=1)
    return np.where(true_class != current_class)

def get_ambiguous_sample_ids(true_label_counts: np.ndarray, threshold: float = 0.30) -> np.ndarray:
    """
    Compute which samples are ambiguous
    :param true_label_counts: The label counts for the true label distribution.
    :param threshold: The label entropy threshold above which a sample is considered ambiguous
    :return: An array with the ids of the ambiguous samples
    """
    label_entropy = compute_label_entropy(true_label_counts)
    return np.where(label_entropy > threshold)[0]


class SimulationStats:
    """
    A class that keeps track of statistics/metrics during the simulation
    """

    def __init__(self, name: str, true_label_counts: np.ndarray, initial_labels: np.ndarray):
        """
        :param name: The name of the simulation
        :param true_label_counts: The label counts for the true label distribution
                                  np.ndarray [num_samples x num_classes]
        :param initial_labels: The initial label counts, np.ndarray [num_samples x num_classes]
        """
        self.name = name
        self.initial_labels = np.copy(initial_labels)
        self.true_label_counts = true_label_counts
        self.true_ambiguous_cases = get_ambiguous_sample_ids(true_label_counts)
        self.true_distribution = true_label_counts / np.sum(true_label_counts, axis=-1, keepdims=True)

        self.selected_sample_id: List[int] = list()
        self.num_fetches: List[int] = list()
        self.accuracy: List[float] = list()
        self.avg_total_variation: List[float] = list()
        self.selection_type: List[SelectionType] = list()
        self.selector_stats: Dict[str, Any] = {key: list() for key in STAT_FIELDS}

        mislabelled_ids_current, ambiguous_case_ids_current = self.get_noisy_and_ambiguous_cases(initial_labels)
        self.mislabelled_not_ambiguous_sample_ids = [mislabelled_ids_current]
        self.mislabelled_ambiguous_sample_ids = [ambiguous_case_ids_current]
        self.num_initial_mislabelled_not_ambiguous = self.mislabelled_not_ambiguous_sample_ids[0].size
        self.num_initial_mislabelled_ambiguous = self.mislabelled_ambiguous_sample_ids[0].size
        self.num_remaining_mislabelled_not_ambiguous: List[int] = list()
        self.num_remaining_mislabelled_ambiguous: List[int] = list()

    def get_noisy_and_ambiguous_cases(self, current_label_counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute which of the current labels are still mislabelled, separate the former into ambiguous and not ambiguous
        samples
        :param current_label_counts: The label counts of the current iteration
        :return: A tuple containing an array with the current mislabelled not ambiguous sample ids and an array with
        the current mislabelled ambiguous sample ids.
        """
        # Find the potential label noise and ambiguous cases
        label_mismatch_ids_current = get_mislabelled_sample_ids(self.true_label_counts, current_label_counts)
        # Split the label mismatch cases into ambiguous and clear label noise types
        mislabelled_ids_current = np.setdiff1d(label_mismatch_ids_current, self.true_ambiguous_cases)
        ambiguous_case_ids_current = np.array(np.intersect1d(label_mismatch_ids_current, self.true_ambiguous_cases))
        return mislabelled_ids_current, ambiguous_case_ids_current

    def record_selector_stats(self, selector_stats: Dict[str, Any]) -> None:
        """
        """
        if len(selector_stats) == 0:
            return

        for key in STAT_FIELDS:
            if key in selector_stats:
                self.selector_stats[key].append(selector_stats[key])

    def record_iteration(self, selected_sample_id: int, num_fetches: int, current_label_counts: np.ndarray) -> None:
        """

        :param selected_sample_id: The sample id that was selected at this iteration
        :param num_fetches: The number of fetches (relabels) it took to achieve a majority
        :param current_label_counts: The labels counts for the current iteration
        :return:
        """
        self.selected_sample_id.append(selected_sample_id)
        self.num_fetches.append(num_fetches)
        self.accuracy.append(compute_accuracy(current_label_counts, self.true_label_counts))
        current_distribution = current_label_counts / np.sum(current_label_counts, axis=-1, keepdims=True)
        self.avg_total_variation.append(np.nanmean(total_variation(self.true_distribution, current_distribution)))

        mislabelled_ids_current, ambiguous_case_ids_current = self.get_noisy_and_ambiguous_cases(current_label_counts)
        mislabelled_ids_prev = self.mislabelled_not_ambiguous_sample_ids[-1]
        ambiguous_case_ids_prev = self.mislabelled_ambiguous_sample_ids[-1]
        selection_type = compute_selection_type_of_current_iter(selected_sample_id,
                                                                self.true_ambiguous_cases,
                                                                self.true_label_counts,
                                                                mislabelled_ids_current, ambiguous_case_ids_current,
                                                                mislabelled_ids_prev, ambiguous_case_ids_prev)
        self.selection_type.append(selection_type)

        self.num_remaining_mislabelled_not_ambiguous.append(len(mislabelled_ids_current))
        self.num_remaining_mislabelled_ambiguous.append(len(ambiguous_case_ids_current))
        self.mislabelled_not_ambiguous_sample_ids.append(mislabelled_ids_current)
        self.mislabelled_ambiguous_sample_ids.append(ambiguous_case_ids_current)

    def log_last_iter(self) -> None:
        """
        Log the statistics of the last iteration
        :return: None
        """

        logging.info(f"Method: {self.name}, selected_id: {self.selected_sample_id[-1]} "
                     f"accuracy: {self.accuracy[-1]}")
        logging.info(f"Remaining label clear noise cases: {self.num_remaining_mislabelled_not_ambiguous[-1]} "
                     f"and ambiguous noise cases: {self.num_remaining_mislabelled_ambiguous[-1]}")


class SimulationStatsDistribution(object):
    """
    A class that takes a list of simulation statistics and creates a distribution over them.
    """

    def __init__(self, simulation_stats_list: List[SimulationStats]):
        """

        :param simulation_stats_list: A list of SimulationStats objects
        """

        self.simulation_stats = simulation_stats_list
        end_point = max([np.max(np.cumsum(sim_stats.num_fetches)) for sim_stats in simulation_stats_list])
        start_point = min([np.min(np.cumsum(sim_stats.num_fetches)) for sim_stats in simulation_stats_list])
        self.num_initial_mislabelled_not_ambiguous = simulation_stats_list[0].num_initial_mislabelled_not_ambiguous
        self.num_initial_mislabelled_ambiguous = simulation_stats_list[0].num_initial_mislabelled_ambiguous
        self.name = simulation_stats_list[0].name
        self.num_fetches = np.arange(start_point, end_point)
        self.accuracy = self._interpolate_and_make_dist_array(self.num_fetches, simulation_stats_list, 'accuracy')
        self.avg_total_variation = self._interpolate_and_make_dist_array(
            self.num_fetches, simulation_stats_list, 'avg_total_variation')

        self.num_remaining_mislabelled_not_ambiguous =\
            self._interpolate_and_make_dist_array(self.num_fetches, simulation_stats_list,
                                                  'num_remaining_mislabelled_not_ambiguous')
        self.num_remaining_mislabelled_ambiguous = \
            self._interpolate_and_make_dist_array(self.num_fetches, simulation_stats_list,
                                                  'num_remaining_mislabelled_ambiguous')

    @staticmethod
    def _interpolate_and_make_dist_array(num_fetches: np.ndarray,
                                         simulation_stats_list: List[SimulationStats],
                                         fp_attr_name: str) -> np.ndarray:
        return np.array([np.interp(num_fetches, np.cumsum(sim_stats.num_fetches),
                        sim_stats.__getattribute__(fp_attr_name)) for sim_stats in simulation_stats_list])


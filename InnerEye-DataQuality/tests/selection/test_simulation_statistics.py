#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
from InnerEyeDataQuality.selection.simulation_statistics import SimulationStats
from InnerEyeDataQuality.selection.simulation_statistics import SelectionType

true_distribution = np.array([[.0, .0, 1.0],
                              [.1, 0.9, .0],
                              [.5, .5, .0],
                              [.5, .3, .2],
                              [1.0, .0, .0]])
initial_labels = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [1, 0, 0],
                          [0, 1, 0],
                          [1, 0, 0]])

def test_get_noisy_and_ambiguous_cases() -> None:

    sim_stats = SimulationStats(name="test_simulation_stats",
                                true_label_counts=true_distribution,
                                initial_labels=initial_labels)

    assert sim_stats.mislabelled_not_ambiguous_sample_ids == [1]
    assert sim_stats.mislabelled_ambiguous_sample_ids == [3]

    current_labels = np.copy(initial_labels)
    current_labels[0, :] = [10, 0, 1]
    mislabelled_cases, ambiguous_cases = sim_stats.get_noisy_and_ambiguous_cases(current_label_counts=current_labels)
    assert np.all(mislabelled_cases == [0, 1])
    assert ambiguous_cases == [3]


def test_record_iteration() -> None:
    sim_stats = SimulationStats(name="test_simulation_stats",
                                true_label_counts=true_distribution,
                                initial_labels=initial_labels)
    current_labels = np.copy(initial_labels)

    # Sample a clear label noise case
    current_labels[1, :] = [1, 1, 0]
    sim_stats.record_iteration(selected_sample_id=1, num_fetches=1, current_label_counts=current_labels)
    assert sim_stats.selection_type[0].value == SelectionType.MISLABELLED_CASE_SELECTED_NOT_CORRECTED.value

    # Sample a clear label noise case
    current_labels[1, :] = [1, 2, 0]
    sim_stats.record_iteration(selected_sample_id=1, num_fetches=1, current_label_counts=current_labels)
    assert sim_stats.selection_type[1].value == SelectionType.MISLABELLED_CASE_SELECTED_CORRECTED.value
    assert np.all(sim_stats.num_remaining_mislabelled_not_ambiguous == [1, 0])

    # Sample an ambiguous case
    current_labels[3, :] = [2, 1, 0]
    sim_stats.record_iteration(selected_sample_id=3, num_fetches=2, current_label_counts=current_labels)
    assert np.all(sim_stats.mislabelled_not_ambiguous_sample_ids[-1] == [])
    assert np.all(sim_stats.mislabelled_ambiguous_sample_ids[-1] == [])
    assert np.all(sim_stats.mislabelled_ambiguous_sample_ids[-2] == [3])

    assert sim_stats.selection_type[2].value == SelectionType.AMBIGUOUS_CASE_SELECTED_CORRECTED.value
    assert np.all(sim_stats.num_remaining_mislabelled_ambiguous == [1, 1, 0])

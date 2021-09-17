#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
from pathlib import Path

import h5py
import numpy as np
from InnerEyeDataQuality.datasets.label_distribution import LabelDistribution
from InnerEyeDataQuality.selection.selectors.base import SampleSelector
from InnerEyeDataQuality.selection.selectors.graph import GraphBasedSelector
from InnerEyeDataQuality.selection.simulation_statistics import SimulationStats
from InnerEyeDataQuality.utils.generic import create_folder


class DataCurationSimulator(object):
    """
    Class that runs the reactive learning simulation given a Selector object
    """

    def __init__(self,
                 initial_labels: np.ndarray,
                 label_distribution: LabelDistribution,
                 relabel_budget: int,
                 sample_selector: SampleSelector,
                 seed: int = 1234,
                 name: str = "Default Simulation") -> None:
        super().__init__()
        """
        """
        self.relabel_budget = relabel_budget
        self.name = name
        self.random_seed = seed
        self.sample_selector = sample_selector
        label_distribution.random_state = np.random.RandomState(seed)

        # Initialise the label pool and first set of labels for the simulation
        self._current_labels = initial_labels
        self._label_distribution = label_distribution
        self._global_stats = SimulationStats(name, label_distribution.label_counts, initial_labels)

    def fetch_until_majority(self, sample_idx: int) -> int:
        """
        Sample labels unitl a majority is formed for a given sample index
        :param sample_idx: The sample index for which the labels will be sampled
        :return:
        """
        majority_formed = False
        num_fetches_per_sample = 0
        while not majority_formed:
            label = self._label_distribution.sample(sample_idx)
            self._current_labels[sample_idx, label] += 1
            _arr = np.sort(self._current_labels[sample_idx])
            majority_formed = _arr[-1] != _arr[-2]
            num_fetches_per_sample += 1
            logging.debug(f"Sample ID: {sample_idx}, Selected label: {label}")
            logging.debug(f"Sample ID: {sample_idx}, Current labels: {self._current_labels[sample_idx]}")

        return num_fetches_per_sample

    def run_simulation(self, plot_samples: bool = False) -> None:
        """
        """
        logging.info(f"Running Simulation Using {self.sample_selector.name} Selector ...")
        num_relabels = 0
        _iter = 0
        while num_relabels <= self.relabel_budget:
            logging.info(f"\nIteration {_iter}")
            sample_id = int(self.sample_selector.get_batch_of_samples_to_annotate(self._current_labels, 1)[0])
            num_fetches = self.fetch_until_majority(sample_id)

            self._global_stats.record_iteration(sample_id, num_fetches, self._current_labels)
            self._global_stats.record_selector_stats(self.sample_selector.stats)
            self._global_stats.log_last_iter()
            num_relabels += num_fetches
            _iter += 1
            if isinstance(self.sample_selector, GraphBasedSelector) and plot_samples:
                self.sample_selector.plot_selected_sample(sample_id, include_knn=False)

            # update beliefs if required
            if self.sample_selector.use_active_relabelling:
                self.sample_selector.update_beliefs(num_relabels)

    @property
    def global_stats(self) -> SimulationStats:
        return self._global_stats

    @property
    def current_labels(self) -> np.ndarray:
        return self._current_labels

    def save_simulator_results(self, output_directory: Path) -> None:
        create_folder(output_directory)
        output_file_path = output_directory / "simulator_output.hdf"

        # Store acquired labels in simulation directory.
        with h5py.File(output_file_path, 'w') as outfile:
            dataset = outfile.create_dataset('current_labels', data=self.current_labels)
            # add meta information
            dataset.attrs['num_relabels'] = self.relabel_budget
            dataset.attrs['selector_name'] = self.name
            dataset.attrs['label_accuracy'] = self._global_stats.accuracy[-1]

    @staticmethod
    def load_simulator_results(hdf_path: Path) -> np.ndarray:
        # load the associated labels from simulation.
        logging.info("")
        logging.info(f"Loading curated labels from HDF5 file: {hdf_path}")
        with h5py.File(hdf_path, 'r') as infile:
            dataset = infile['current_labels']
            current_labels = dataset[...]

            logging.info(f"Number of relabels: {dataset.attrs['num_relabels']}")
            logging.info(f"Selector used in relabelling: {dataset.attrs['selector_name']}")
            logging.info(f"Label accuracy: {dataset.attrs['label_accuracy']}")

        return current_labels

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from azureml.core import Run

from InnerEye.Azure.azure_util import RUN_CONTEXT, download_outputs_from_run, fetch_child_runs, tag_values_all_distinct
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME, check_properties_are_not_none
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, \
    create_recovery_checkpoint_path, get_best_checkpoint_path
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER, DeepLearningConfig


@dataclass(frozen=True)
class RunRecovery:
    """
    Class to encapsulate information relating to run recovery (eg: check point paths for parent and child runs)
    """
    checkpoints_roots: List[Path]

    @staticmethod
    def download_best_checkpoints_from_child_runs(config: DeepLearningConfig, run: Run) -> RunRecovery:
        """
        Downloads the best checkpoints from all child runs of the provided Hyperdrive parent run.
        The checkpoints for the sibling runs will go into folder 'OTHER_RUNS/<cross_validation_split>'
        in the checkpoint folder. There is special treatment for the child run that is equal to the present AzureML
        run, its checkpoints will be read off the checkpoint folder as-is.
        :param config: Model related configs.
        :param run: The Hyperdrive parent run to download from.
        :return: run recovery information
        """
        child_runs: List[Run] = fetch_child_runs(run)
        if not child_runs:
            raise ValueError(f"AzureML run {run.id} does not have any child runs.")
        logging.info(f"Run {run.id} has {len(child_runs)} child runs: {', '.join(c.id for c in child_runs)}")
        tag_to_use = 'cross_validation_split_index'
        can_use_split_indices = tag_values_all_distinct(child_runs, tag_to_use)
        # download checkpoints for the child runs in the root of the parent
        child_runs_checkpoints_roots: List[Path] = []
        for child in child_runs:
            if child.id == RUN_CONTEXT.id:
                # We expect to find the file(s) we need in config.checkpoint_folder
                child_dst = config.checkpoint_folder
            else:
                subdir = str(child.tags[tag_to_use] if can_use_split_indices else child.number)
                child_dst = config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME / subdir
                download_outputs_from_run(
                    blobs_path=Path(CHECKPOINT_FOLDER) / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
                    destination=child_dst,
                    run=child,
                    is_file=True
                )
            child_runs_checkpoints_roots.append(child_dst)
        return RunRecovery(checkpoints_roots=child_runs_checkpoints_roots)

    @staticmethod
    def download_all_checkpoints_from_run(config: DeepLearningConfig, run: Run) -> RunRecovery:
        """
        Downloads all checkpoints of the provided run: The best checkpoint and the recovery checkpoint.
        A single folder inside the checkpoints folder will be created that contains the downloaded checkpoints.
        :param config: Model related configs.
        :param run: Run whose checkpoints should be recovered
        :return: run recovery information
        """
        if fetch_child_runs(run):
            raise ValueError(f"AzureML run {run.id} has child runs, this method does not support those.")

        root_output_dir = config.checkpoint_folder / run.id
        download_outputs_from_run(
            blobs_path=Path(CHECKPOINT_FOLDER),
            destination=root_output_dir,
            run=run
        )
        return RunRecovery(checkpoints_roots=[root_output_dir])

    def get_recovery_checkpoint_paths(self) -> List[Path]:
        return [create_recovery_checkpoint_path(x) for x in self.checkpoints_roots]

    def get_best_checkpoint_paths(self) -> List[Path]:
        return [get_best_checkpoint_path(x) for x in self.checkpoints_roots]

    def _validate(self) -> None:
        check_properties_are_not_none(self)
        if len(self.checkpoints_roots) == 0:
            raise ValueError("checkpoints_roots must not be empty")

    def __post_init__(self) -> None:
        self._validate()
        logging.info(f"Storing {len(self.checkpoints_roots)}checkpoints roots:")
        for p in self.checkpoints_roots:
            logging.info(str(p))

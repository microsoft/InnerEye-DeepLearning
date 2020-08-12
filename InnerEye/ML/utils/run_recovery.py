#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from azureml.core import Run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import RUN_CONTEXT, fetch_child_runs, fetch_run, get_cross_validation_split_index, \
    is_cross_validation_child_run
from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.common import create_checkpoint_path
from InnerEye.ML.deep_learning_config import CHECKPOINT_FOLDER
from InnerEye.ML.model_config_base import ModelConfigBase


@dataclass(frozen=True)
class RunRecovery:
    """
    Class to encapsulate information relating to run recovery (eg: check point paths for parent and child runs)
    """
    checkpoints_roots: List[Path]

    @staticmethod
    def download_checkpoints_from_recovery_run(azure_config: AzureConfig,
                                               config: ModelConfigBase,
                                               run_context: Optional[Run] = None) -> RunRecovery:
        """
        Downloads checkpoints of run corresponding to the run_recovery_id in azure_config, and any
        checkpoints of the child runs if they exist.

        :param azure_config: Azure related configs.
        :param config: Model related configs.
        :param run_context: Context of the current run (will be used to find the target AML workspace)
        :return:RunRecovery
        """
        run_context = run_context or RUN_CONTEXT
        workspace = azure_config.get_workspace()

        # Find the run to recover in AML workspace
        if not azure_config.run_recovery_id:
            raise ValueError("A valid run_recovery_id is required to download recovery checkpoints, found None")

        run_to_recover = fetch_run(workspace, azure_config.run_recovery_id.strip())
        # Handle recovery of a HyperDrive cross validation run (from within a successor HyperDrive run,
        # not in ensemble creation). In this case, run_recovery_id refers to the parent prior run, so we
        # need to set run_to_recover to the child of that run whose split index is the same as that of
        # the current (child) run.
        if is_cross_validation_child_run(run_context):
            run_to_recover = next(x for x in fetch_child_runs(run_to_recover) if
                                  get_cross_validation_split_index(x) == get_cross_validation_split_index(run_context))

        return RunRecovery.download_checkpoints_from_run(azure_config, config, run_to_recover)

    @staticmethod
    def download_checkpoints_from_run(azure_config: AzureConfig,
                                      config: ModelConfigBase,
                                      run: Run) -> RunRecovery:
        """
        Downloads checkpoints of the provided run or, if applicable, its children.
        :param azure_config: Azure related configs.
        :param config: Model related configs.
        :param run: Run whose checkpoints should be recovered
        :return: run recovery information
        """
        child_runs: List[Run] = fetch_child_runs(run)
        logging.debug(f"Run has ID {run.id} and initial child runs are:")
        for child_run in child_runs:
            logging.debug(f"     {child_run.id}")
        root_output_dir = Path(config.checkpoint_folder) / run.id
        # download checkpoints for the run
        azure_config.download_outputs_from_run(
            blobs_path=Path(CHECKPOINT_FOLDER),
            destination=root_output_dir,
            run=run
        )
        if len(child_runs) > 0:
            # download checkpoints for the child runs in the root of the parent
            child_runs_checkpoints_roots: List[Path] = []
            for child in child_runs:
                child_dst = root_output_dir / str(child.number)
                azure_config.download_outputs_from_run(
                    blobs_path=Path(CHECKPOINT_FOLDER),
                    destination=child_dst,
                    run=child
                )
                child_runs_checkpoints_roots.append(child_dst)
            return RunRecovery(checkpoints_roots=child_runs_checkpoints_roots)
        else:
            return RunRecovery(checkpoints_roots=[root_output_dir])

    def get_checkpoint_paths(self, epoch: int, for_mean_teacher_model: bool = False) -> List[Path]:
        return [create_checkpoint_path(x, epoch, for_mean_teacher_model) for x in self.checkpoints_roots]

    def _validate(self) -> None:
        check_properties_are_not_none(self)
        if len(self.checkpoints_roots) == 0:
            raise ValueError("checkpoints_roots must not be empty")

    def __post_init__(self) -> None:
        self._validate()
        logging.info(f"Recovering from checkpoints roots: {self.checkpoints_roots}")

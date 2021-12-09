#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.deep_learning_config import get_best_checkpoint_path


@dataclass(frozen=True)
class RunRecovery:
    """
    Class to encapsulate information relating to run recovery (eg: check point paths for parent and child runs)
    """
    checkpoints_roots: List[Path]

    def get_recovery_checkpoint_paths(self) -> List[Path]:
        from InnerEye.ML.utils.checkpoint_handling import get_recovery_checkpoint_path
        return [get_recovery_checkpoint_path(x) for x in self.checkpoints_roots]

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

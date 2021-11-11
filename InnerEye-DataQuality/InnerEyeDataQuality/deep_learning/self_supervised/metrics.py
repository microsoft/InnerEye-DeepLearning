#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from pytorch_lightning.metrics.functional import auroc


class AreaUnderRocCurve(Metric):
    """
    Computes the area under the receiver operating curve (ROC).
    """

    def __init__(self) -> None:
        super().__init__(dist_sync_on_step=False)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)
        self.name = "auc"

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:  # type: ignore
        assert preds.dim() == 2 and targets.dim() == 1 and \
               preds.shape[1] == 2 and preds.shape[0] == targets.shape[
                   0], f"Expected 2-dim preds, 1-dim targets, but got: preds = {preds.shape}, targets = {targets.shape}"
        self.preds.append(preds)  # type: ignore
        self.targets.append(targets)  # type: ignore

    def compute(self) -> torch.Tensor:
        """
        Computes a metric from the stored predictions and targets.
        """
        preds = torch.cat(self.preds)  # type: ignore
        targets = torch.cat(self.targets)  # type: ignore
        if torch.unique(targets).numel() == 1:
            return torch.tensor(np.nan)
        return auroc(preds[:, 1], targets)

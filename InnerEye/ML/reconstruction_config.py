#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any

from InnerEye.ML.model_config_base import ModelConfigBase

class ReconstructionModelBase(ModelConfigBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
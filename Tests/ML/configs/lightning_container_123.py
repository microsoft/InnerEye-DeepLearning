#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.ML.lightning_container import LightningContainer


class DummyLightningContainer(LightningContainer):
    def __init__(self):
        super().__init__()

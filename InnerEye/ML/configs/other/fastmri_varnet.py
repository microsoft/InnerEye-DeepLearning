#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
from fastMRI.fastmri.pl_modules import VarNetModule


class FastMriVarNetDemoModel(LightningWithInference,
                             VarNetModule):
    pass


class FastMriVarnetDemoContainer(LightningContainer):
    pass

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.common_util import add_folder_to_sys_path_if_needed
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference

add_folder_to_sys_path_if_needed("fastMRI")

from fastmri.pl_modules import VarNetModule


class VarNetWithInference(LightningWithInference,
                          VarNetModule):
    pass


class FastMriDemoContainer(LightningContainer):

    def create_lightning_module(self) -> LightningWithInference:
        return VarNetWithInference()

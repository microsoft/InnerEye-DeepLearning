#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.ML.configs.segmentation.BasicModel2Epochs import BasicModel2Epochs


class BasicModel2Epochs1Channel(BasicModel2Epochs):
    def __init__(self) -> None:
        super().__init__(image_channels=["ct"])

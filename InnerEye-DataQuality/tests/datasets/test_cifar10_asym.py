#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from default_paths import CIFAR10_ROOT_DIR
from InnerEyeDataQuality.datasets.cifar10_asym_noise import CIFAR10AsymNoise

def test_cifar10_asym() -> None:
    CIFAR10AsymNoise(root=str(CIFAR10_ROOT_DIR), train=True, transform=None, download=True)

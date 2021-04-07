#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.SSL.lightning_containers.ssl_container import SSLDatasetName
from InnerEye.SSL.lightning_containers.ssl_image_classifier import SSLClassifierContainer


class SSLClassifierCIFAR(SSLClassifierContainer):
    def __init__(self, **kwargs):
        super().__init__(
            classifier_dataset_name=SSLDatasetName.CIFAR10,
            random_seed=1,
            recovery_checkpoint_save_interval=5,
            num_epochs=100,
            l_rate=1e-4,
            num_workers=6)

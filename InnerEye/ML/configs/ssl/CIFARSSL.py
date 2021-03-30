#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.SSL.lightning_containers.ssl_base import EncoderName, SSLContainer, SSLType


class CIFARSimCLR(SSLContainer):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="CIFAR10",
                         num_workers=6,
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2500,
                         batch_size=512,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_type=SSLType.SimCLR)

class CIFARBYOL(SSLContainer):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="CIFAR10",
                         num_workers=6,
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2500,
                         batch_size=512,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_type=SSLType.BYOL)
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.SSL.lightning_containers.ssl_base import EncoderName, SSLContainer, SSLDatasetName, SSLType


class CIFAR10SimCLR(SSLContainer):
    """
    This module trains an SSL encoder using SimCLR on CIFAR10 and finetunes a linear head on CIFAR10 too.
    """
    def __init__(self, **kwargs):
        super().__init__(dataset_name=SSLDatasetName.CIFAR10,
            linear_head_dataset_name=SSLDatasetName.CIFAR10,
            random_seed=1,
            recovery_checkpoint_save_interval=200,
            num_epochs=2500,
            batch_size=512,
            num_workers=6,
            ssl_encoder=EncoderName.resnet50,
            ssl_type=SSLType.SimCLR)

class CIFAR10BYOL(SSLContainer):
    """
    This module trains an SSL encoder using BYOL on CIFAR10 and finetunes a linear head on CIFAR10 too.
    """
    def __init__(self, **kwargs):
        super().__init__(dataset_name=SSLDatasetName.CIFAR10,
            linear_head_dataset_name=SSLDatasetName.CIFAR10,
            random_seed=1,
            recovery_checkpoint_save_interval=200,
            num_epochs=2500,
            batch_size=512,
            num_workers=6,
            ssl_encoder=EncoderName.resnet50,
            ssl_type=SSLType.BYOL)

class CIFAR10CIFAR100BYOL(SSLContainer):
    """
    This module trains an SSL encoder using BYOL on CIFAR10 and finetunes a linear head on CIFAR100.
    """
    def __init__(self, **kwargs):
        super().__init__(dataset_name=SSLDatasetName.CIFAR10,
            linear_head_dataset_name=SSLDatasetName.CIFAR100,
            random_seed=1,
            recovery_checkpoint_save_interval=200,
            num_epochs=2500,
            batch_size=512,
            num_workers=6,
            ssl_encoder=EncoderName.resnet50,
            ssl_type=SSLType.BYOL)
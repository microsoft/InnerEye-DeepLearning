#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.SSL.lightning_containers.ssl_base import EncoderName, SSLContainer, SSLType


class RSNAKaggleBYOL(SSLContainer):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="RSNAKaggle",
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2000,
                         batch_size=1600,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_type=SSLType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         path_augmentation_config=repository_root_directory() / "InnerEye" / "ML" / "configs" / "ssl" /
                                                  "rsna_augmentations.yaml")


class RSNAKaggleSimCLR(SSLContainer):
    def __init__(self, **kwargs):
        super().__init__(dataset_name="RSNAKaggle",
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2000,
                         batch_size=1600,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_type=SSLType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         path_augmentation_config=repository_root_directory() / "InnerEye" / "ML" / "configs" / "ssl" /
                                                  "rsna_augmentations.yaml")

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from InnerEye.SSL.lightning_containers.ssl_image_classifier import SSLClassifierContainer
from InnerEye.SSL.utils import SSLType

RSNA_AZURE_DATASET_ID = "rsna_pneumonia_detection_kaggle_dataset"
NIH_AZURE_DATASET_ID = "nih-training-set"

path_encoder_augmentation_cxr = repository_root_directory() / "InnerEye" / "ML" / "configs" / "ssl" / \
                                "cxr_encoder_augmentations.yaml"

path_linear_head_augmentation_cxr = repository_root_directory() / "InnerEye" / "ML" / "configs" / \
                                                        "ssl" / "cxr_linear_head.yaml"


class RSNA_RSNA_BYOL(SSLContainer):
    """
    Config to train SSL model on RSNA Pneumonia detection Challenge dataset and use the same dataset to finetune
    linear head on top,
    """

    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.RSNAKaggle,
                         classifier_dataset_name=SSLDatasetName.RSNAKaggle,
                         azure_dataset_id=RSNA_AZURE_DATASET_ID,
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2000,
                         ssl_training_batch_size=1200,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_training_path_augmentation_config=path_encoder_augmentation_cxr)


class NIH_RSNA_BYOL(SSLContainer):
    """
    Config to train SSL model on NIH ChestXray dataset and use the RSNA Pneumonia detection Challenge dataset to
    finetune the linear head on top for performance monitoring.
    """

    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIH,
                         classifier_dataset_name=SSLDatasetName.RSNAKaggle,
                         azure_dataset_id=NIH_AZURE_DATASET_ID,
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2000,
                         ssl_training_batch_size=1200,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_training_path_augmentation_config=path_encoder_augmentation_cxr,
                         extra_azure_dataset_ids=[RSNA_AZURE_DATASET_ID],
                         classifier_augmentations_path=path_linear_head_augmentation_cxr)


class NIH_RSNA_SimCLR(SSLContainer):
    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIH,
                         classifier_dataset_name=SSLDatasetName.RSNAKaggle,
                         azure_dataset_id=NIH_AZURE_DATASET_ID,
                         random_seed=1,
                         recovery_checkpoint_save_interval=200,
                         num_epochs=2000,
                         ssl_training_batch_size=1200,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_training_path_augmentation_config=path_encoder_augmentation_cxr,
                         extra_azure_dataset_ids=[RSNA_AZURE_DATASET_ID],
                         classifier_augmentations_path=path_linear_head_augmentation_cxr)


class CXRImageClassifier(SSLClassifierContainer):
    def __init__(self) -> None:
        super().__init__(classifier_dataset_name=SSLDatasetName.RSNAKaggle,
                         random_seed=1,
                         recovery_checkpoint_save_interval=10,
                         num_epochs=200,
                         use_balanced_binary_loss_for_linear_head=True,
                         extra_azure_dataset_ids=[RSNA_AZURE_DATASET_ID],
                         classifier_augmentations_path=path_linear_head_augmentation_cxr)

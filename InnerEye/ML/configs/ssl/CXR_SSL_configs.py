#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from InnerEye.ML.SSL.lightning_containers.ssl_image_classifier import SSLClassifierContainer
from InnerEye.ML.SSL.utils import SSLTrainingType

RSNA_AZURE_DATASET_ID = "rsna_pneumonia_detection_kaggle_dataset"
NIH_AZURE_DATASET_ID = "nih-training-set"

path_encoder_augmentation_cxr = repository_root_directory() / "InnerEye" / "ML" / "configs" / "ssl" / \
                                "cxr_ssl_encoder_augmentations.yaml"

path_linear_head_augmentation_cxr = repository_root_directory() / "InnerEye" / "ML" / "configs" / \
                                    "ssl" / "cxr_linear_head_augmentations.yaml"

class NIH_RSNA_BYOL(SSLContainer):
    """
    Config to train SSL model on NIHCXR ChestXray dataset and use the RSNA Pneumonia detection Challenge dataset to
    finetune the linear head on top for performance monitoring.
    """

    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         azure_dataset_id=NIH_AZURE_DATASET_ID,
                         random_seed=1,
                         num_epochs=1000,
                         # We usually train this model with 16 GPUs, giving an effective batch size of 1200
                         ssl_training_batch_size=75,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         extra_azure_dataset_ids=[RSNA_AZURE_DATASET_ID],
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)

class NIH_RSNA_SimCLR(SSLContainer):
    def __init__(self) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         azure_dataset_id=NIH_AZURE_DATASET_ID,
                         random_seed=1,
                         num_epochs=1000,
                         # We usually train this model with 16 GPUs, giving an effective batch size of 1200
                         ssl_training_batch_size=75,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         extra_azure_dataset_ids=[RSNA_AZURE_DATASET_ID],
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)


class CXRImageClassifier(SSLClassifierContainer):
    def __init__(self) -> None:
        super().__init__(linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                         random_seed=1,
                         num_epochs=200,
                         use_balanced_binary_loss_for_linear_head=True,
                         azure_dataset_id=RSNA_AZURE_DATASET_ID,
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr)

from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from InnerEye.ML.SSL.utils import SSLTrainingType
from InnerEye.ML.configs.ssl.CXR_SSL_configs import NIH_AZURE_DATASET_ID, path_encoder_augmentation_cxr, \
    path_linear_head_augmentation_cxr

COVID_DATASET_ID = "id-of-your-dataset"


class NIH_COVID_BYOL(SSLContainer):
    """
    Class to train a SSL model on NIH dataset and monitor embeddings quality on a Covid Dataset.
    """

    def __init__(self,
                 covid_dataset_id=COVID_DATASET_ID,
                 **kwargs):
        super().__init__(ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                         linear_head_dataset_name=SSLDatasetName.Covid,
                         random_seed=1,
                         recovery_checkpoint_save_interval=50,
                         recovery_checkpoints_save_last_k=3,
                         num_epochs=500,
                         ssl_training_batch_size=1200,  # This runs  with 16 gpus (4 nodes)
                         num_workers=12,
                         ssl_encoder=EncoderName.densenet121,
                         ssl_training_type=SSLTrainingType.BYOL,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=path_encoder_augmentation_cxr,
                         extra_azure_dataset_ids=[covid_dataset_id],
                         azure_dataset_id=NIH_AZURE_DATASET_ID,
                         linear_head_augmentation_config=path_linear_head_augmentation_cxr,
                         online_evaluator_lr=1e-5,
                         linear_head_batch_size=64,
                         **kwargs)

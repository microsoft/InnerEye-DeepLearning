from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sys

from InnerEye.ML.configs.histo_configs.ssl.HistoSimCLRContainer import HistoSSLContainer
from InnerEye.ML.SSL.datamodules_and_datasets.histopathology.tcgacrck_tiles_dataset import (
    TcgaCrck_TilesDatasetWithReturnIndex)
from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from InnerEye.ML.SSL.utils import SSLTrainingType


current_file = Path(__file__)
print(f"Running container from {current_file}")
print(f"Sys path container level {sys.path}")

local_mode = False
path_local_data: Optional[Path]
if local_mode:
    is_debug_model = True
    drop_last = False
    # This dataset has been used for test purposes on a local machine, change to your local path
    path_local_data = Path("/tmp/datasets/TCGA-CRCk")
    num_workers = 0
    AZURE_DATASET_ID = 'Dummy'
    num_epochs = 2
else:
    is_debug_model = False
    drop_last = False
    path_local_data = None
    num_workers = 12
    AZURE_DATASET_ID = 'TCGA-CRCk'
    num_epochs = 200


class SSLDatasetNameRadiomicsNN(SSLDatasetName, Enum):
    TCGA_CRCK = "CRCKTilesDataset"


class CRCK_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on CRCK tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.

    It has been tested locally and on AML on the full training dataset (93408 tiles).
    """
    SSLContainer._SSLDataClassMappings.update({SSLDatasetNameRadiomicsNN.TCGA_CRCK.value:
                                                   TcgaCrck_TilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetNameRadiomicsNN.TCGA_CRCK,
                         linear_head_dataset_name=SSLDatasetNameRadiomicsNN.TCGA_CRCK,
                         local_dataset=path_local_data,
                         azure_dataset_id=AZURE_DATASET_ID,
                         random_seed=1,
                         num_workers=num_workers,
                         is_debug_model=is_debug_model,
                         recovery_checkpoint_save_interval=50,
                         recovery_checkpoints_save_last_k=3,
                         num_epochs=num_epochs,
                         ssl_training_batch_size=48,  # GPU memory is at 70% with batch_size=32, 2GPUs
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=None,  # Change to path_augmentation to use the config
                         linear_head_augmentation_config=None,  # Change to path_augmentation to use the config
                         drop_last=drop_last,
                         **kwargs)

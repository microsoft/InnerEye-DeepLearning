from enum import Enum
from pathlib import Path
from typing import Any, Optional
import sys
from InnerEye.Common.fixed_paths import repository_root_directory

from InnerEye.ML.SSL.lightning_containers.ssl_container import EncoderName, SSLContainer, SSLDatasetName
from InnerEye.ML.SSL.utils import SSLTrainingType
from InnerEye.ML.SSL.datamodules_and_datasets.histopathology.panda_tiles_dataset import (
    PandaTilesDatasetWithReturnIndex)
from InnerEye.ML.configs.histo_configs.ssl.HistoSimCLRContainer import HistoSSLContainer


current_file = Path(__file__)
print(f"Running container from {current_file}")
print(f"Sys path container level {sys.path}")

path_augmentation = repository_root_directory() / "InnerEyePrivate" / "ML" / "configs" / "histo_configs" / "ssl" / \
                    "panda_encoder_augmentations.yml"

local_mode = True
path_local_data: Optional[Path]
if local_mode:
    is_debug_model = True
    drop_last = False
    # This dataset has been used for test purposes on a local machine, change to your local path
    path_local_data = Path("/Users/vsalvatelli/workspace/data/PANDA_tiles_toy/panda_tiles_level1_224")
    num_workers = 0
    PANDA_AZURE_DATASET_ID = 'Dummy'
else:
    is_debug_model = False
    drop_last = False
    path_local_data = None
    num_workers = 12
    PANDA_AZURE_DATASET_ID = 'panda_tiles_level1_224'


class SSLDatasetNameRadiomicsNN(SSLDatasetName, Enum):
    PANDA = "PandaTilesDataset"


class PANDA_SimCLR(HistoSSLContainer):
    """
    Config to train SSL model on Panda tiles dataset.
    Augmentation can be configured by using a configuration yml file or by specifying the set of transformations
    in the _get_transforms method.

    It has been tested on a toy local dataset (2 slides) and on AML on (~25 slides).
    """
    SSLContainer._SSLDataClassMappings.update({SSLDatasetNameRadiomicsNN.PANDA.value: PandaTilesDatasetWithReturnIndex})

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(ssl_training_dataset_name=SSLDatasetNameRadiomicsNN.PANDA,
                         linear_head_dataset_name=SSLDatasetNameRadiomicsNN.PANDA,
                         local_dataset=path_local_data,
                         azure_dataset_id=PANDA_AZURE_DATASET_ID,
                         random_seed=1,
                         num_workers=num_workers,
                         is_debug_model=is_debug_model,
                         recovery_checkpoint_save_interval=50,
                         recovery_checkpoints_save_last_k=3,
                         num_epochs=200,
                         ssl_training_batch_size=32,
                         ssl_encoder=EncoderName.resnet50,
                         ssl_training_type=SSLTrainingType.SimCLR,
                         use_balanced_binary_loss_for_linear_head=True,
                         ssl_augmentation_config=None,  # Change to path_augmentation to use the config
                         linear_head_augmentation_config=None,  # Change to path_augmentation to use the config
                         drop_last=drop_last,
                         **kwargs)

""" PandaImageNetMIL is a LightningContainer for multiple instance learning
Run using python InnerEyePrivate/ML/runner.py --model=PandaImageNetMIL
"""
from pathlib import Path

from torchvision.models import resnet18

from health_ml.data.histopathology.datamodules.panda_module import PandaTilesDataModule
from health_ml.data.histopathology.datasets.default_paths import (PANDA_TILES_DATASET_DIR,
                                                                   PANDA_TILES_DATASET_ID)
from health_ml.data.histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from health_ml.models.histopathology.attention_layers import GatedAttentionLayer
from health_ml.models.histopathology.deepmil import DeepMILModule
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.SSL.encoders import ImageNetEncoder


class PandaImageNetMIL(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        root_path_str = PANDA_TILES_DATASET_DIR
        self.local_dataset = Path(root_path_str)
        self.num_epochs = 100
        self.max_bag_size = 1000
        self.batch_size = 8
        self.azure_dataset_id = PANDA_TILES_DATASET_ID
        self.use_dataset_mount = True

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        return DeepMILModule(encoder=ImageNetEncoder(feature_extraction_model=resnet18,
                                                     tile_size=224, n_channels=3),
                             label_column=PandaTilesDataset.LABEL_COLUMN,
                             n_classes=PandaTilesDataset.N_CLASSES,
                             pooling_layer=GatedAttentionLayer,
                             class_weights=self.data_module.class_weights)

    def get_data_module(self) -> PandaTilesDataModule:
        return PandaTilesDataModule(root_path=self.local_dataset,
                                    max_bag_size=self.max_bag_size,
                                    batch_size=self.batch_size)

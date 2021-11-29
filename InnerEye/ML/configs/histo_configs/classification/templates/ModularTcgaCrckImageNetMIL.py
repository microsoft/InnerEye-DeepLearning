""" A version of TcgaCrckImageNetMIL with a standalone encoder object instead of a DeepMIL subclass
Uses TCGA-CRCk dataset and loads tiles of the same slide (bags) together for MIL
Run using python InnerEyePrivate/ML/runner.py --model=ModularTcgaCrckImageNetMIL
"""
from pathlib import Path

from torchvision.models import resnet18

from health_ml.data.histopathology.datamodules.base_module import TilesDataModule
from health_ml.data.histopathology.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from health_ml.models.histopathology.deepmil import DeepMILModule
from health_ml.models.histopathology.attention_layers import GatedAttentionLayer
from health_ml.data.histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.SSL.encoders  import ImageNetEncoder, TileEncoder


class ModularTcgaCrckImageNetMIL(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        root_path_str = "/tmp/datasets/TCGA-CRCk"
        self.local_dataset = Path(root_path_str)
        self.num_epochs = 16
        self.max_bag_size = 1000
        self.azure_dataset_id = "TCGA-CRCk"
        self.use_dataset_mount = True

        self.encoder = self.get_encoder()

    def get_encoder(self) -> TileEncoder:
        return ImageNetEncoder(feature_extraction_model=resnet18,
                               tile_size=224, n_channels=3)

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        return DeepMILModule(encoder=self.encoder,
                             label_column=TcgaCrck_TilesDataset.LABEL_COLUMN,
                             n_classes=1,
                             pooling_layer=GatedAttentionLayer,
                             class_weights=self.data_module.class_weights)

    def get_data_module(self) -> TilesDataModule:
        return TcgaCrckTilesDataModule(root_path=self.local_dataset,
                                       max_bag_size=self.max_bag_size)

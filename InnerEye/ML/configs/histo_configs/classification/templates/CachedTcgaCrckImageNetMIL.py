""" A version of TcgaCrckImageNetMIL that enables caching the extracted features
Uses TCGA-CRCk dataset and loads tiles of the same slide (bags) together for MIL
Run using python InnerEyePrivate/ML/runner.py --model=CachedTcgaCrckImageNetMIL
"""
from pathlib import Path

from monai.transforms import Compose
from torchvision.models import resnet18

from InnerEye.ML.lightning_container import LightningContainer
from health_ml.data.histopathology.datamodules.base_module import CacheMode, TilesDataModule
from health_ml.data.histopathology.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from InnerEyePrivate.Histopathology.models.attention_layers import GatedAttentionLayer
from InnerEyePrivate.Histopathology.models.deepmil import DeepMILModule
from InnerEyePrivate.Histopathology.models.encoders import IdentityEncoder, ImageNetEncoder, TileEncoder
from InnerEyePrivate.Histopathology.models.transforms import EncodeTilesBatchd, LoadTilesBatchd
from health_ml.data.histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset


class CachedTcgaCrckImageNetMIL(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        root_path_str = "/tmp/datasets/TCGA-CRCk"
        self.local_dataset = Path(root_path_str)
        self.num_epochs = 16
        self.max_bag_size = 3
        self.batch_size = 8
        self.azure_dataset_id = "TCGA-CRCk"
        self.use_dataset_mount = True

        self.cache_mode = CacheMode.MEMORY  # or CacheMode.DISK
        self.save_precache = True
        # Data will be pre-cached here for use by all DDP processes
        self.cache_dir = Path("/tmp/innereye_cache/CachedTcgaCrckImageNetMIL/")

        self.encoder = self.get_encoder()
        self.encoder.cuda()

    def get_encoder(self) -> TileEncoder:
        return ImageNetEncoder(feature_extraction_model=resnet18,
                               tile_size=224, n_channels=3)

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        return DeepMILModule(encoder=IdentityEncoder(input_dim=(self.encoder.num_encoding,)),
                             label_column=TcgaCrck_TilesDataset.LABEL_COLUMN,
                             n_classes=1,
                             pooling_layer=GatedAttentionLayer,
                             class_weights=self.data_module.class_weights)

    def get_data_module(self) -> TilesDataModule:
        image_key = TcgaCrck_TilesDataset.IMAGE_COLUMN
        transform = Compose([LoadTilesBatchd(image_key, progress=True),
                             EncodeTilesBatchd(image_key, self.encoder)])
        return TcgaCrckTilesDataModule(root_path=self.local_dataset,
                                       max_bag_size=self.max_bag_size,
                                       batch_size=self.batch_size,
                                       transform=transform,
                                       cache_mode=self.cache_mode,
                                       save_precache=self.save_precache,
                                       cache_dir=self.cache_dir)

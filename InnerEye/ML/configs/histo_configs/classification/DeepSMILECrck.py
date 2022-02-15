#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""DeepSMILECrck is the container for experiments relating to DeepSMILE using the TCGA-CRCk dataset.
Run using `python InnerEyePrivate/ML/runner.py --model=DeepSMILECrck --encoder_type=<encoder class name>`

For convenience, this module also defines encoder-specific containers that can be invoked without
additional arguments, e.g. `python InnerEyePrivate/ML/runner.py --model=TcgaCrckImageNetMIL`

Reference:
- Schirris (2021). DeepSMILE: Self-supervised heterogeneity-aware multiple instance learning for DNA
damage response defect classification directly from H&E whole-slide images. arXiv:2107.09405
"""
from typing import Any, List
from pathlib import Path
import os
from monai.transforms import Compose
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from health_azure.utils import CheckpointDownloader
from health_azure.utils import get_workspace
from health_ml.networks.layers.attention_layers import AttentionLayer
from InnerEye.Common import fixed_paths
from InnerEye.ML.Histopathology.datamodules.base_module import CacheMode, CacheLocation
from InnerEye.ML.Histopathology.datamodules.base_module import TilesDataModule
from InnerEye.ML.Histopathology.datamodules.tcga_crck_module import TcgaCrckTilesDataModule
from InnerEye.ML.common import get_best_checkpoint_path

from InnerEye.ML.Histopathology.models.transforms import (
    EncodeTilesBatchd,
    LoadTilesBatchd,
)
from InnerEye.ML.Histopathology.models.encoders import (
    HistoSSLEncoder,
    ImageNetEncoder,
    ImageNetSimCLREncoder,
    InnerEyeSSLEncoder,
)
from InnerEye.ML.configs.histo_configs.classification.BaseMIL import BaseMIL
from InnerEye.ML.Histopathology.datasets.tcga_crck_tiles_dataset import TcgaCrck_TilesDataset


class DeepSMILECrck(BaseMIL):
    def __init__(self, **kwargs: Any) -> None:
        # Define dictionary with default params that can be overriden from subclasses or CLI
        default_kwargs = dict(
            # declared in BaseMIL:
            pooling_type=AttentionLayer.__name__,
            encoding_chunk_size=60,
            cache_mode=CacheMode.MEMORY,
            precache_location=CacheLocation.CPU,
            # declared in DatasetParams:
            local_dataset=Path("/tmp/datasets/TCGA-CRCk"),
            azure_dataset_id="TCGA-CRCk",
            # To mount the dataset instead of downloading in AML, pass --use_dataset_mount in the CLI
            # declared in TrainerParams:
            num_epochs=50,
            # declared in WorkflowParams:
            number_of_cross_validation_splits=5,
            cross_validation_split_index=0,
            # declared in OptimizerParams:
            l_rate=5e-4,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99),
        )
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)

        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = (
            self.best_checkpoint_filename + ".ckpt"
        )
        self.checkpoint_folder_path = "outputs/checkpoints/"

        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_folder_path,
            monitor="val/auroc",
            filename=self.best_checkpoint_filename,
            auto_insert_metric_name=False,
            mode="max",
        )
        self.callbacks = best_checkpoint_callback

    @property
    def cache_dir(self) -> Path:
        return Path(
            f"/tmp/innereye_cache1/{self.__class__.__name__}-{self.encoder_type}/"
        )

    def setup(self) -> None:
        if self.encoder_type == InnerEyeSSLEncoder.__name__:
            from InnerEye.ML.configs.histo_configs.run_ids import innereye_ssl_checkpoint_crck_4ws
            self.downloader = CheckpointDownloader(
                azure_config_json_path=get_workspace(),
                run_id=innereye_ssl_checkpoint_crck_4ws,
                checkpoint_filename="last.ckpt",
                download_dir="outputs/",
                remote_checkpoint_dir=Path("outputs/checkpoints")
            )
            os.chdir(fixed_paths.repository_parent_directory())
            self.downloader.download_checkpoint_if_necessary()

        self.encoder = self.get_encoder()
        self.encoder.cuda()
        self.encoder.eval()

    def get_data_module(self) -> TilesDataModule:
        image_key = TcgaCrck_TilesDataset.IMAGE_COLUMN
        transform = Compose(
            [
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder),
            ]
        )
        return TcgaCrckTilesDataModule(
            root_path=self.local_dataset,
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            transform=transform,
            cache_mode=self.cache_mode,
            precache_location=self.precache_location,
            cache_dir=self.cache_dir,
            number_of_cross_validation_splits=self.number_of_cross_validation_splits,
            cross_validation_split_index=self.cross_validation_split_index,
        )

    def get_callbacks(self) -> List[Callback]:
        return super().get_callbacks() + [self.callbacks]

    def get_path_to_best_checkpoint(self) -> Path:
        """
        Returns the full path to a checkpoint file that was found to be best during training, whatever criterion
        was applied there.
        """
        # absolute path is required for registering the model.
        absolute_checkpoint_path = Path(fixed_paths.repository_root_directory(),
                                        self.checkpoint_folder_path,
                                        self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path.is_file():
            return absolute_checkpoint_path

        absolute_checkpoint_path_parent = Path(fixed_paths.repository_parent_directory(),
                                    self.checkpoint_folder_path,
                                    self.best_checkpoint_filename_with_suffix)
        if absolute_checkpoint_path_parent.is_file():
            return absolute_checkpoint_path_parent

        checkpoint_path = get_best_checkpoint_path(Path(self.checkpoint_folder_path))
        if checkpoint_path.is_file():
            return checkpoint_path

        raise ValueError("Path to best checkpoint not found")


class TcgaCrckImageNetMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class TcgaCrckImageNetSimCLRMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class TcgaCrckInnerEyeSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=InnerEyeSSLEncoder.__name__, **kwargs)


class TcgaCrckHistoSSLMIL(DeepSMILECrck):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)

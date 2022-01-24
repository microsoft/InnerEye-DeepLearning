#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List
from pathlib import Path
import os
from monai.transforms import Compose
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import Callback

from health_azure.utils import CheckpointDownloader
from health_azure.utils import get_workspace, is_running_in_azure_ml
from health_ml.networks.layers.attention_layers import GatedAttentionLayer
from InnerEye.Common import fixed_paths
from InnerEye.ML.Histopathology.datamodules.panda_module import PandaTilesDataModule
from InnerEye.ML.Histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
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
    IdentityEncoder
)
from InnerEye.ML.configs.histo_configs.classification.BaseMIL import BaseMIL
from InnerEye.ML.Histopathology.datasets.panda_dataset import PandaDataset
from InnerEye.ML.Histopathology.models.deepmil import DeepMILModule


class DeepSMILEPanda(BaseMIL):
    def __init__(self, **kwargs: Any) -> None:
        default_kwargs = dict(
            # declared in BaseMIL:
            pooling_type=GatedAttentionLayer.__name__,
            # declared in DatasetParams:
            local_dataset=Path("/tmp/datasets/PANDA_tiles"),
            azure_dataset_id="PANDA_tiles",
            extra_azure_dataset_ids=["PANDA"],
            extra_local_dataset_paths=[Path("/tmp/datasets/PANDA")],
            # To mount the dataset instead of downloading in AML, pass --use_dataset_mount in the CLI
            # declared in TrainerParams:
            num_epochs=200,
            # use_mixed_precision = True,
            # declared in WorkflowParams:
            number_of_cross_validation_splits=5,
            cross_validation_split_index=0,
            # declared in OptimizerParams:
            l_rate=5e-3,
            weight_decay=1e-4,
            adam_betas=(0.9, 0.99))
        default_kwargs.update(kwargs)
        super().__init__(**default_kwargs)
        super().__init__(**default_kwargs)
        if not is_running_in_azure_ml():
            self.num_epochs = 1
        self.best_checkpoint_filename = "checkpoint_max_val_auroc"
        self.best_checkpoint_filename_with_suffix = (
            self.best_checkpoint_filename + ".ckpt"
        )
        self.checkpoint_folder_path = "outputs/checkpoints/"
        best_checkpoint_callback = ModelCheckpoint(
            dirpath=self.checkpoint_folder_path,
            monitor="val/accuracy",
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
            from InnerEye.ML.configs.histo_configs.run_ids import innereye_ssl_checkpoint_binary
            self.downloader = CheckpointDownloader(
                aml_workspace=get_workspace(),
                run_id=innereye_ssl_checkpoint_binary,  # innereye_ssl_checkpoint
                checkpoint_filename="best_checkpoint.ckpt",  # "last.ckpt",
                download_dir="outputs/",
                remote_checkpoint_dir=Path("outputs/checkpoints")
            )
            os.chdir(fixed_paths.repository_parent_directory())
            self.downloader.download_checkpoint_if_necessary()
        self.encoder = self.get_encoder()
        self.encoder.cuda()
        self.encoder.eval()

    def get_data_module(self) -> PandaTilesDataModule:
        image_key = PandaTilesDataset.IMAGE_COLUMN
        transform = Compose(
            [
                LoadTilesBatchd(image_key, progress=True),
                EncodeTilesBatchd(image_key, self.encoder),
            ]
        )
        return PandaTilesDataModule(
            root_path=self.local_dataset,
            max_bag_size=self.max_bag_size,
            batch_size=self.batch_size,
            transform=transform,
            cache_mode=self.cache_mode,
            save_precache=self.save_precache,
            cache_dir=self.cache_dir,
            number_of_cross_validation_splits=self.number_of_cross_validation_splits,
            cross_validation_split_index=self.cross_validation_split_index,
        )

    def create_model(self) -> DeepMILModule:
        self.data_module = self.get_data_module()
        # Encoding is done in the datamodule, so here we provide instead a dummy
        # no-op IdentityEncoder to be used inside the model
        self.slide_dataset = self.get_slide_dataset()
        self.level = 1
        return DeepMILModule(encoder=IdentityEncoder(input_dim=(self.encoder.num_encoding,)),
                             label_column=self.data_module.train_dataset.LABEL_COLUMN,
                             n_classes=self.data_module.train_dataset.N_CLASSES,
                             pooling_layer=self.get_pooling_layer(),
                             class_weights=self.data_module.class_weights,
                             l_rate=self.l_rate,
                             weight_decay=self.weight_decay,
                             adam_betas=self.adam_betas,
                             slide_dataset=self.get_slide_dataset(),
                             tile_size=self.tile_size,
                             level=self.level)

    def get_slide_dataset(self) -> PandaDataset:                                                
        return PandaDataset(root=self.extra_local_dataset_paths[0])                             # type: ignore

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


class PandaImageNetMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetEncoder.__name__, **kwargs)


class PandaImageNetSimCLRMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=ImageNetSimCLREncoder.__name__, **kwargs)


class PandaInnerEyeSSLMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=InnerEyeSSLEncoder.__name__, **kwargs)


class PandaHistoSSLMIL(DeepSMILEPanda):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(encoder_type=HistoSSLEncoder.__name__, **kwargs)

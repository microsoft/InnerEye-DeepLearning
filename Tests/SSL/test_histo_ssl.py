import os
from typing import Tuple, Type

import pytest
import torch

from health_ml.data.histopathology.datasets.default_paths import PANDA_TILES_DATASET_DIR, TCGA_CRCK_DATASET_DIR
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import InnerEyeVisionDataModule
from InnerEye.ML.SSL.datamodules_and_datasets.dataset_cls_utils import InnerEyeDataClassBaseWithReturnIndex
from InnerEye.ML.SSL.datamodules_and_datasets.histopathology.panda_tiles_dataset import PandaTilesDatasetWithReturnIndex
from InnerEye.ML.SSL.datamodules_and_datasets.histopathology.tcgacrck_tiles_dataset import (
    TcgaCrck_TilesDatasetWithReturnIndex)
from InnerEye.ML.configs.histo_configs.ssl.HistoSimCLRContainer import HistoSSLContainer

DATASET_DIRS = {
    TcgaCrck_TilesDatasetWithReturnIndex: TCGA_CRCK_DATASET_DIR,
    PandaTilesDatasetWithReturnIndex: PANDA_TILES_DATASET_DIR,
}


def _test_tiles_ssl_datamodule(dataset_cls: Type[InnerEyeDataClassBaseWithReturnIndex],
                               dataset_sizes: Tuple[int, int, int]) -> None:
    batch_size = 5
    train_transforms, val_transforms = HistoSSLContainer()._get_transforms(augmentation_config=None,
                                                                           dataset_name='',
                                                                           is_ssl_encoder_module=True)
    data_module = InnerEyeVisionDataModule(dataset_cls=dataset_cls,
                                           data_dir=DATASET_DIRS[dataset_cls],
                                           val_split=0.1,
                                           return_index=False,
                                           train_transforms=train_transforms,
                                           val_transforms=val_transforms,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           drop_last=True)
    data_module.prepare_data()
    data_module.setup()
    assert len(data_module.dataset_train) == dataset_sizes[0]
    assert len(data_module.dataset_val) == dataset_sizes[1]
    assert len(data_module.dataset_test) == dataset_sizes[2]

    def validate_batch(batch: Tuple) -> None:
        (images_v1, images_v2), labels = batch
        assert isinstance(images_v1, torch.Tensor)
        assert isinstance(images_v2, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert images_v1.shape == images_v2.shape == (batch_size, 3, 224, 224)
        assert labels.shape == (batch_size,)

    training_batch = next(iter(data_module.train_dataloader()))
    validate_batch(training_batch)

    validation_batch = next(iter(data_module.val_dataloader()))
    validate_batch(validation_batch)


@pytest.mark.skipif(not os.path.isdir(TCGA_CRCK_DATASET_DIR),
                    reason="TCGA-CRCk dataset is unavailable")
def test_tcga_crck_ssl_datamodule2() -> None:
    _test_tiles_ssl_datamodule(TcgaCrck_TilesDatasetWithReturnIndex,
                               dataset_sizes=(84068, 9340, 98904))


@pytest.mark.skipif(not os.path.isdir(PANDA_TILES_DATASET_DIR),
                    reason="Panda tiles dataset is unavailable")
def test_panda_ssl_datamodule() -> None:
    _test_tiles_ssl_datamodule(PandaTilesDatasetWithReturnIndex,
                               dataset_sizes=(615158, 68350, 683508))

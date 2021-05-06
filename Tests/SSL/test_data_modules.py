#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Dict

import PIL
import numpy as np
import pytest
import torch
from pytorch_lightning.trainer.supporters import CombinedLoader

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.SSL.datamodules_and_datasets.cifar_datasets import InnerEyeCIFAR10
from InnerEye.ML.SSL.datamodules_and_datasets.cxr_datasets import RSNAKaggleCXR
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import CombinedDataModule, InnerEyeVisionDataModule
from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import InnerEyeCIFARLinearHeadTransform, \
    InnerEyeCIFARTrainTransform, get_cxr_ssl_transforms
from InnerEye.ML.SSL.lightning_containers.ssl_container import SSLContainer, SSLDatasetName
from InnerEye.ML.SSL.utils import SSLDataModuleType, load_ssl_augmentation_config
from InnerEye.ML.configs.ssl.CXR_SSL_configs import path_encoder_augmentation_cxr
from Tests.SSL.test_ssl_containers import _create_test_cxr_data

path_to_test_dataset = full_ml_test_data_path("cxr_test_dataset")
_create_test_cxr_data(path_to_test_dataset)
cxr_augmentation_config = load_ssl_augmentation_config(path_encoder_augmentation_cxr)


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_weights_innereye_module() -> None:
    """
    Tests if weights in CXR data module are correctly initialized
    """
    transforms = get_cxr_ssl_transforms(cxr_augmentation_config,
                                        return_two_views_per_sample=True)
    data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                           return_index=False,
                                           train_transforms=transforms[0],
                                           val_transforms=transforms[1],
                                           data_dir=str(path_to_test_dataset),
                                           batch_size=1,
                                           seed=1,
                                           num_workers=0)
    data_module.setup()
    class_weights = data_module.compute_class_weights()
    assert class_weights is not None
    assert torch.isclose(class_weights, torch.tensor([0.21, 0.79], dtype=torch.float32), atol=1e-3).all()
    assert len(data_module.dataset_train) == 240
    assert len(data_module.dataset_val) == 60
    training_batch = next(iter(data_module.train_dataloader()))
    # Assert we have two images and one label given the InnerEyeCIFARTrainTransform
    images, labels = training_batch
    images_v1, images_v2 = images
    assert images_v1.shape == images_v2.shape == torch.Size([1, 3, 224, 224])


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_innereye_vision_module() -> None:
    """
    Test properties of loaded CIFAR datasets via InnerEyeVisionDataModule.
    Tests as well that the transforms return data in the expected type and shape.
    :return:
    """
    data_module = InnerEyeVisionDataModule(dataset_cls=InnerEyeCIFAR10,
                                           val_split=0.1,
                                           return_index=False,
                                           train_transforms=InnerEyeCIFARTrainTransform(32),
                                           val_transforms=InnerEyeCIFARLinearHeadTransform(32),
                                           data_dir=None,
                                           batch_size=5,
                                           shuffle=False,
                                           num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    assert len(data_module.dataset_train) == 45000
    assert len(data_module.dataset_val) == 5000
    assert len(data_module.dataset_test) == 10000

    training_batch = next(iter(data_module.train_dataloader()))
    # Assert we have two images and one label given the InnerEyeCIFARTrainTransform
    images, labels = training_batch
    images_v1, images_v2 = images
    assert images_v1.shape == images_v2.shape == torch.Size([5, 3, 32, 32])
    assert labels.tolist() == [6, 2, 8, 6, 6]

    validation_batch = next(iter(data_module.val_dataloader()))
    # Assert we have one image and one label given the InnerEyeCIFARLinearHeadTransform
    images_v1, labels = validation_batch
    assert images_v1.shape == torch.Size([5, 3, 32, 32])
    assert labels.tolist() == [7, 9, 1, 5, 7]


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_innereye_vision_datamodule_with_return_index() -> None:
    """
    Tests that the return index flag, modifies __getitem__ as expected i.e.
    returns the index on top of the transformed image and label.
    """
    data_module = InnerEyeVisionDataModule(dataset_cls=InnerEyeCIFAR10,
                                           return_index=True,
                                           train_transforms=InnerEyeCIFARLinearHeadTransform(32),
                                           val_transforms=None,
                                           data_dir=None,
                                           batch_size=5,
                                           shuffle=False,
                                           num_workers=0)
    data_module.prepare_data()
    data_module.setup()
    training_batch = next(iter(data_module.train_dataloader()))
    # Assert we have one one index, one image and one label given the InnerEyeCIFARLinearHeadTransform
    indices, images, labels = training_batch
    assert images.shape == torch.Size([5, 3, 32, 32])
    assert indices.tolist() == [37542, 44491, 216, 43688, 41558]
    assert labels.tolist() == [6, 2, 8, 6, 6]


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_get_transforms_in_SSL_container_for_cxr_data() -> None:
    """
    Tests that the internal _get_transforms function returns data of the expected type of CXR.
    Tests that is_ssl_encoder_module induces the correct type of transform pipeline (dual vs single view).
    """
    test_container = SSLContainer(linear_head_dataset_name=SSLDatasetName.RSNAKaggleCXR,
                                  ssl_training_dataset_name=SSLDatasetName.NIHCXR,
                                  ssl_augmentation_config=path_encoder_augmentation_cxr)
    test_container._load_config()
    dual_view_transform, _ = test_container._get_transforms(augmentation_config=test_container.ssl_augmentation_params,
                                                            dataset_name=SSLDatasetName.NIHCXR.value,
                                                            is_ssl_encoder_module=True)

    test_img = PIL.Image.fromarray(np.ones([312, 312]) * 255.).convert("L")
    v1, v2 = dual_view_transform(test_img)
    # Images should be cropped to 224 x 224 and expanded to 3 channels according to config
    assert v1.shape == v2.shape == torch.Size([3, 224, 224])
    # The three channels should simply by duplicates
    assert (v1[0] == v1[1]).all() and (v1[1] == v1[2]).all()
    # Two returned images should be different
    assert (v1 != v2).any()

    single_view_transform, _ = test_container._get_transforms(
        augmentation_config=test_container.ssl_augmentation_params,
        dataset_name=SSLDatasetName.NIHCXR.value,
        is_ssl_encoder_module=False)
    v1 = single_view_transform(test_img)
    # Images should be cropped to 224 x 224 and expanded to 3 channels according to config
    assert v1.shape == torch.Size([3, 224, 224])


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_get_transforms_in_SSL_container_for_cifar_data() -> None:
    """
    Tests that the internal _get_transforms function returns data of the expected type of CXR.
    Tests that is_ssl_encoder_module induces the correct type of transform pipeline (dual vs single view).
    """
    test_container = SSLContainer()
    dual_view_transform, _ = test_container._get_transforms(augmentation_config=None,
                                                            dataset_name=SSLDatasetName.CIFAR10.value,
                                                            is_ssl_encoder_module=True)
    img_array_with_black_square = np.ones([32, 32, 3], dtype=np.uint8)
    img_array_with_black_square[10:20, 10:20, :] = 255
    test_img = PIL.Image.fromarray(img_array_with_black_square)
    v1, v2 = dual_view_transform(test_img)
    # Images not be resized
    assert v1.shape == v2.shape == torch.Size([3, 32, 32])
    # Two returned images should be different
    assert (v1 != v2).any()

    single_view_transform, _ = test_container._get_transforms(augmentation_config=None,
                                                              dataset_name=SSLDatasetName.CIFAR10.value,
                                                              is_ssl_encoder_module=False)
    v1 = single_view_transform(test_img)
    # Images should be cropped to 224 x 224 and expanded to 3 channels according to config
    assert v1.shape == torch.Size([3, 32, 32])


@pytest.mark.skipif(is_windows(), reason="Too slow on windows")
def test_combined_data_module() -> None:
    """
    Tests the behavior of CombinedDataModule
    """
    _, val_transform = get_cxr_ssl_transforms(cxr_augmentation_config,
                                              return_two_views_per_sample=False)

    # Datamodule expected to have 12 training batches - 3 val
    long_data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                                val_split=0.2,
                                                return_index=True,
                                                train_transforms=None,
                                                val_transforms=val_transform,
                                                data_dir=str(path_to_test_dataset),
                                                # 300 images in total in test dataset
                                                batch_size=20,
                                                shuffle=False)
    long_data_module.setup()
    # Datamodule expected to have 4 training batches - 1 val
    short_data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                                 val_split=0.2,
                                                 return_index=True,
                                                 train_transforms=None,
                                                 val_transforms=val_transform,
                                                 data_dir=str(path_to_test_dataset),
                                                 # 300 images in total in test dataset
                                                 batch_size=60,
                                                 shuffle=False)
    short_data_module.setup()

    combined_loader = CombinedDataModule(encoder_module=long_data_module,
                                         linear_head_module=short_data_module,
                                         use_balanced_loss_linear_head=False)

    assert combined_loader.num_classes == 2
    # num samples has to return number of training samples in encoder data.
    assert combined_loader.num_samples == 240
    # we don't want to compute the weights for balanced loss
    assert combined_loader.class_weights is None

    # Check the behavior if we want to compute the weights for balanced loss
    combined_loader = CombinedDataModule(encoder_module=long_data_module,
                                         linear_head_module=short_data_module,
                                         use_balanced_loss_linear_head=True)
    assert combined_loader.class_weights is not None
    assert torch.isclose(combined_loader.class_weights,
                         torch.tensor([0.21, 0.79], dtype=torch.float32), atol=1e-3).all()

    # PyTorch Lightning expects a dictionary of loader at training time.
    # It will take care of "filling in the gaps" automatically in the trainer
    # (i.e. matching the length of the shortest dataloader to the length of the
    # longest).
    train_dataloaders = combined_loader.train_dataloader()
    assert isinstance(train_dataloaders, Dict)
    assert len(train_dataloaders[SSLDataModuleType.ENCODER]) == 12
    assert len(train_dataloaders[SSLDataModuleType.LINEAR_HEAD]) == 4

    # Weirdly, in PL the handling of combined loader is different in validation
    # stage. There, it is expected to return an object of type "CombinedDataLoader" that
    # takes care of the aggregation of batches.
    indices_classifier_module_short = []
    val_dataloader = combined_loader.val_dataloader()
    assert isinstance(val_dataloader, CombinedLoader)
    for batch in val_dataloader:
        assert set(batch.keys()) == {SSLDataModuleType.ENCODER, SSLDataModuleType.LINEAR_HEAD}
        indices_classifier_module_short.append(tuple(batch[SSLDataModuleType.LINEAR_HEAD][0].tolist()))
    # Check that combined dataloader fills in the shorter datamodule to match the number of batches of the longest one
    # by looping from the beginning again.
    assert len(val_dataloader) == 3
    assert len(set(indices_classifier_module_short)) == 1

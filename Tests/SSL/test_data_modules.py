#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import PIL
import numpy as np
import torch

from InnerEye.ML.SSL.datamodules_and_datasets.cifar_datasets import InnerEyeCIFAR10
from InnerEye.ML.SSL.datamodules_and_datasets.cxr_datasets import RSNAKaggleCXR
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import InnerEyeVisionDataModule
from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import InnerEyeCIFARLinearHeadTransform, \
    InnerEyeCIFARTrainTransform
from InnerEye.ML.SSL.lightning_containers.ssl_container import SSLContainer, SSLDatasetName
from InnerEye.ML.configs.ssl.CXR_SSL_configs import path_encoder_augmentation_cxr
from Tests.SSL.test_ssl_containers import _create_test_cxr_data
from Tests.utils_for_tests import full_ml_test_data_path

path_to_test_dataset = full_ml_test_data_path("cxr_test_dataset")
_create_test_cxr_data(path_to_test_dataset)


def test_weights_innereye_module() -> None:
    """
    Tests if weights in CXR data module are correctly initialized
    """

    data_module = InnerEyeVisionDataModule(dataset_cls=RSNAKaggleCXR,
                                           return_index=True,
                                           train_transforms=None,
                                           val_transforms=None,
                                           data_dir=str(path_to_test_dataset),
                                           batch_size=1,
                                           seed=1)
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
    assert images_v1.shape == images_v2.shape == torch.Size([1, 3, 256, 256])


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
                                           seed=1,
                                           shuffle=False)
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
    assert labels.tolist() == [0, 1, 6, 3, 6]

    validation_batch = next(iter(data_module.val_dataloader()))
    # Assert we have one image and one label given the InnerEyeCIFARLinearHeadTransform
    images_v1, labels = validation_batch
    assert images_v1.shape == torch.Size([5, 3, 32, 32])
    assert labels.tolist() == [6, 0, 2, 3, 3]


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
                                           seed=1,
                                           shuffle=False)
    data_module.prepare_data()
    data_module.setup()
    training_batch = next(iter(data_module.train_dataloader()))
    # Assert we have one one index, one image and one label given the InnerEyeCIFARLinearHeadTransform
    indices, images, labels = training_batch
    assert images.shape == torch.Size([5, 3, 32, 32])
    assert indices.tolist() == [45845, 11799, 43880, 43701, 41303]
    assert labels.tolist() == [0, 1, 6, 3, 6]


def test_get_transforms_in_SSL_container_for_cxr_data() -> None:
    """
    Tests that the internal _get_transforms function returns data of the expected type of CXR.
    Tests that is_ssl_encoder_module induces the correct type of transform pipeline (dual vs single view).
    """
    test_container = SSLContainer(classifier_dataset_name=SSLDatasetName.RSNAKaggle,
                                  ssl_training_dataset_name=SSLDatasetName.NIH,
                                  ssl_augmentation_config=path_encoder_augmentation_cxr)
    test_container._load_config()
    dual_view_transform, _ = test_container._get_transforms(augmentation_config=test_container.ssl_augmentation_params,
                                                            dataset_name=SSLDatasetName.NIH.value,
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
        dataset_name=SSLDatasetName.NIH.value,
        is_ssl_encoder_module=False)
    v1 = single_view_transform(test_img)
    # Images should be cropped to 224 x 224 and expanded to 3 channels according to config
    assert v1.shape == torch.Size([3, 224, 224])

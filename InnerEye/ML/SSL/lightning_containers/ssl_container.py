#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import param
from pytorch_lightning import LightningModule

from InnerEye.ML.SSL.augmentation_config_utils.config_node import ConfigNode
from InnerEye.ML.SSL.datamodules_and_datasets.cifar_datasets import InnerEyeCIFAR10, InnerEyeCIFAR100
from InnerEye.ML.SSL.datamodules_and_datasets.cxr_datasets import CheXpert, NIH, RSNAKaggleCXR
from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import CombinedDataModule, InnerEyeVisionDataModule
from InnerEye.ML.SSL.datamodules_and_datasets.transforms_utils import InnerEyeCIFARLinearHeadTransform, \
    InnerEyeCIFARTrainTransform, \
    InnerEyeCIFARValTransform, get_cxr_ssl_transforms
from InnerEye.ML.SSL.encoders import get_encoder_output_dim
from InnerEye.ML.SSL.lightning_modules.byol.byol_module import BYOLInnerEye
from InnerEye.ML.SSL.lightning_modules.simclr_module import SimCLRInnerEye
from InnerEye.ML.SSL.lightning_modules.ssl_online_evaluator import SSLOnlineEvaluatorInnerEye
from InnerEye.ML.SSL.utils import SSLDataModuleType, SSLTrainingType, load_ssl_model_config
from InnerEye.ML.lightning_container import LightningContainer


@dataclass
class DataModuleArgs:
    augmentation_params: Optional[ConfigNode]
    dataset_name: str
    dataset_path: Optional[Path]
    batch_size: int


class EncoderName(Enum):
    resnet18 = "resnet18"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    densenet121 = "densenet121"


class SSLDatasetName(Enum):
    RSNAKaggle = "RSNAKaggle"
    NIH = "NIH"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    CheXpert = "CheXpert"


InnerEyeDataModuleTypes = Union[InnerEyeVisionDataModule, CombinedDataModule]


class SSLContainer(LightningContainer):
    """
    This container is the based module to train an SSL model (either using BYOL or SimCLR).
    See Readme for more extensive documentation about its configuration.
    """
    _SSLDataClassMappings = {SSLDatasetName.RSNAKaggle.value: RSNAKaggleCXR,
                             SSLDatasetName.NIH.value: NIH,
                             SSLDatasetName.CIFAR10.value: InnerEyeCIFAR10,
                             SSLDatasetName.CIFAR100.value: InnerEyeCIFAR100,
                             SSLDatasetName.CheXpert.value: CheXpert}

    ssl_augmentation_config = param.ClassSelector(class_=Path, allow_None=True,
                                                  doc="The path to the yaml config defining the parameters of the "
                                                      "augmentations. Ignored for CIFAR10 example")
    ssl_training_dataset_name = param.ClassSelector(class_=SSLDatasetName, doc="The name of the dataset")
    ssl_training_batch_size = param.Integer(
        doc="Total training batch size, will be divided across the number of gpus used for training. For example: if "
            "you specify ssl_training_batch_size=1600 and use 4 nodes with 4 gpus each (i.e. total of 16 GPUs), "
            "the code will provide a per-gpu batch size of 100")
    ssl_training_type = param.ClassSelector(class_=SSLTrainingType, doc="Which algorithm to use for SSL training")
    ssl_encoder = param.ClassSelector(class_=EncoderName, doc="Which encoder to use for SSL")
    use_balanced_binary_loss_for_linear_head = param.Boolean(default=False,
                                                             doc="Whether to use a balanced loss for the training of "
                                                                 "the linear head")
    num_workers = param.Integer(default=6, doc="Number of workers to use for dataloader processes.")
    is_debug_model = param.Boolean(default=False,
                                   doc="If True, the training will be restricted to 1 batch per epoch."
                                       "Used for debugging and tests.")
    classifier_augmentation_config = param.ClassSelector(class_=Path,
                                                         doc="The path to the yaml config for the linear head "
                                                             "augmentations")
    classifier_dataset_name = param.ClassSelector(class_=SSLDatasetName,
                                                  doc="Name of the dataset to use for the linear head training")
    classifier_batch_size = param.Integer(default=256, doc="Batch size for linear head tuning")
    online_evaluator_lr = param.Number(default=1e-4, doc="Learning rate for linear head training during SSL training.")

    def setup(self) -> None:
        from InnerEye.ML.SSL.lightning_containers.ssl_image_classifier import SSLClassifierContainer
        self.total_num_gpus = self.num_gpus_per_node * self.num_nodes
        self._load_config()
        # If you're using the same data for training and linear head, allow the user to specify the dataset only
        # once. Or if you are doing just finetuning of linear head, the user should be able to specify dataset via
        # azure_dataset_id/local_dataset instead of extra_dataset fields (as in this case we only use one dataset).
        if ((self.classifier_dataset_name == self.ssl_training_dataset_name) or isinstance(self,
                                                                                           SSLClassifierContainer)) \
                and len(self.extra_local_dataset_paths) == 0 and self.local_dataset is not None:
            self.extra_local_dataset_paths = [self.local_dataset]
        self.datamodule_args = {SSLDataModuleType.LINEAR_HEAD:
                                    DataModuleArgs(augmentation_params=self.classifier_augmentation_params,
                                                   dataset_name=self.classifier_dataset_name.value,
                                                   dataset_path=self.extra_local_dataset_paths[0] if len(
                                                       self.extra_local_dataset_paths) > 0 else None,
                                                   batch_size=self.classifier_batch_size)}
        if self.ssl_training_dataset_name is not None:
            self.datamodule_args.update(
                {SSLDataModuleType.ENCODER: DataModuleArgs(augmentation_params=self.ssl_augmentation_params,
                                                           dataset_name=self.ssl_training_dataset_name.value,
                                                           dataset_path=self.local_dataset,
                                                           batch_size=self.ssl_training_batch_size)})
        self.data_module: InnerEyeDataModuleTypes = self.get_data_module()
        self.perform_validation_and_test_set_inference = False
        if self.number_of_cross_validation_splits > 1:
            raise NotImplementedError("Cross-validation logic is not implemented for this module.")

    def _load_config(self) -> None:
        # For Chest-XRay you need to specify the parameters of the augmentations via a config file.
        self.ssl_augmentation_params = load_ssl_model_config(
            self.ssl_augmentation_config) if self.ssl_augmentation_config is not None \
            else None
        self.classifier_augmentation_params = load_ssl_model_config(
            self.classifier_augmentation_config) if self.classifier_augmentation_config is not None else \
            self.ssl_augmentation_params

    def create_model(self) -> LightningModule:
        """
        This method must create the actual Lightning model that will be trained.
        """
        # For small images like CIFAR, if using a resnet encoder, switch the first conv layer to a 3x3 kernel instead
        # of a 7x7 conv layer.
        use_7x7_first_conv_in_resnet = False if self.ssl_training_dataset_name.value.startswith("CIFAR") else True
        if self.ssl_training_type == SSLTrainingType.SimCLR:
            model: LightningModule = SimCLRInnerEye(encoder_name=self.ssl_encoder.value,
                                                    dataset_name=self.ssl_training_dataset_name.value,
                                                    use_7x7_first_conv_in_resnet=use_7x7_first_conv_in_resnet,
                                                    gpus=self.total_num_gpus,
                                                    num_samples=self.data_module.num_samples,
                                                    batch_size=self.data_module.batch_size,
                                                    lr=self.l_rate,
                                                    max_epochs=self.num_epochs)
        elif self.ssl_training_type == SSLTrainingType.BYOL:
            model = BYOLInnerEye(encoder_name=self.ssl_encoder.value,
                                 num_samples=self.data_module.num_samples,
                                 batch_size=self.data_module.batch_size,
                                 learning_rate=self.l_rate,
                                 use_7x7_first_conv_in_resnet=use_7x7_first_conv_in_resnet,
                                 warmup_epochs=10)
        else:
            raise ValueError(
                f"Unknown value for ssl_training_type, should be {SSLTrainingType.SimCLR.value} or "
                f"{SSLTrainingType.BYOL.value}. "
                f"Found {self.ssl_training_type.value}")
        model.hparams.update({'ssl_type': self.ssl_training_type.value,
                              "num_classes": self.data_module.num_classes})
        self.encoder_output_dim = get_encoder_output_dim(model, self.data_module)

        return model

    def get_data_module(self) -> InnerEyeDataModuleTypes:
        """
        Gets the data that is used for the training and validation steps.
        Here we use different data loader for training of linear head and training of SSL model.
        """
        if hasattr(self, "data_module"):
            return self.data_module
        encoder_module = self._create_ssl_data_modules(linear_head_module=False)
        linear_head_module = self._create_ssl_data_modules(linear_head_module=True)
        return CombinedDataModule(encoder_module, linear_head_module, self.use_balanced_binary_loss_for_linear_head)

    def _create_ssl_data_modules(self, linear_head_module: bool) -> InnerEyeVisionDataModule:
        """
        Returns torch lightning data module for encoder or linear head
        """
        datamodule_args = self.datamodule_args[SSLDataModuleType.LINEAR_HEAD] if linear_head_module else \
            self.datamodule_args[
                SSLDataModuleType.ENCODER]

        train_transforms, val_transforms = self._get_transforms(datamodule_args.augmentation_params,
                                                                datamodule_args.dataset_name,
                                                                linear_head_module)
        batch_size_per_gpu = datamodule_args.batch_size // self.total_num_gpus if self.total_num_gpus > 0 else \
            datamodule_args.batch_size
        logging.info(f"Batch size per gpu: {batch_size_per_gpu}")
        dm = InnerEyeVisionDataModule(dataset_cls=self._SSLDataClassMappings[datamodule_args.dataset_name],
                                      return_index=linear_head_module,
                                      train_transforms=train_transforms,
                                      val_split=0.1,
                                      val_transforms=val_transforms,
                                      data_dir=str(datamodule_args.dataset_path),
                                      batch_size=batch_size_per_gpu,
                                      num_workers=self.num_workers,
                                      seed=self.random_seed)
        dm.prepare_data()
        dm.setup('fit')
        return dm

    def _get_transforms(self, augmentation_config: Optional[ConfigNode],
                        dataset_name: str,
                        linear_head_module: bool) -> Tuple[Any, Any]:
        if dataset_name in [SSLDatasetName.RSNAKaggle.value, SSLDatasetName.NIH.value, SSLDatasetName.CheXpert.value]:
            assert augmentation_config is not None
            train_transforms, val_transforms = get_cxr_ssl_transforms(augmentation_config, linear_head_module)
        elif dataset_name in [SSLDatasetName.CIFAR10.value, SSLDatasetName.CIFAR100.value]:
            train_transforms = \
                InnerEyeCIFARTrainTransform(32) if not linear_head_module else InnerEyeCIFARLinearHeadTransform(32)
            val_transforms = \
                InnerEyeCIFARValTransform(32) if not linear_head_module else InnerEyeCIFARLinearHeadTransform(32)
        else:
            raise ValueError(f"Dataset {dataset_name} unknown.")

        return train_transforms, val_transforms

    def get_trainer_arguments(self) -> Dict[str, Any]:
        self.online_eval = SSLOnlineEvaluatorInnerEye(class_weights=self.data_module.class_weights,  # type: ignore
                                                      z_dim=self.encoder_output_dim,
                                                      num_classes=self.data_module.num_classes,  # type: ignore
                                                      dataset=self.classifier_dataset_name.value,  # type: ignore
                                                      drop_p=0.2,
                                                      learning_rate=self.online_evaluator_lr)
        trainer_kwargs: Dict[str, Any] = {"callbacks": self.online_eval}
        if self.is_debug_model:
            trainer_kwargs.update({"limit_train_batches": 1, "limit_val_batches": 1})
        return trainer_kwargs

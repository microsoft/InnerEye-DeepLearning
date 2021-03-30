#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from pathlib import Path

import param
from pytorch_lightning import LightningDataModule

from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
from InnerEye.SSL.byol.byol_module import WrapBYOLInnerEye
from InnerEye.SSL.config_node import ConfigNode
from InnerEye.SSL.datamodules.chestxray_datamodule import RSNAKaggleDataModule
from InnerEye.SSL.datamodules.cifar_ie_datamodule import CIFARIEDataModule
from InnerEye.SSL.simclr_module import WrapSimCLRInnerEye

from InnerEye.SSL.ssl_classifier_module import SSLOnlineEvaluatorInnerEye, get_encoder_output_dim
from InnerEye.SSL.utils import load_ssl_model_config


class SSLType(Enum):
    SimCLR = "SimCLR"
    BYOL = "BYOL"


class EncoderName(Enum):
    resnet18 = "resnet18"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    densenet121 = "densenet121"


class SSLContainer(LightningContainer):
    """
    This container is the module to use to train an SSL model either BYOL or SimCLR.
    See Readme for more extensive documentation about its configuration.
    """
    path_augmentation_config = param.ClassSelector(class_=Path, allow_None=True,
                                                   doc="The path to the yaml config defining the parameters of the "
                                                       "augmentations. Ignored for CIFAR10 example")
    dataset_name = param.String(doc="The name of the dataset. Choice between CIFAR10 or RSNAKaggle.")
    batch_size = param.Integer(doc="Total training batch size, splitted across the number of gpus.")
    learning_rate = param.Number(default=1e-3, doc="Learning rate for SSL training")
    ssl_type = param.ClassSelector(class_=SSLType, doc="Which algorithm to use for SSL training")
    ssl_encoder = param.ClassSelector(class_=EncoderName, doc="Which encoder to use for SSL")
    use_balanced_binary_loss_for_linear_head = param.Boolean(default=False,
                                                             doc="Whether to use a balanced loss for the training of "
                                                                 "the linear head")
    num_workers = param.Integer(default=6, doc="Number of workers to use for dataloader processes.")
    debug = param.Boolean(default=False,
                          doc="If True, the training will be restricted to 2 batches per epoch. Used for debugging "
                              "and tests.")

    def setup(self):
        self._load_config()
        self.data_module = self.get_data_module()
        self.perform_validation_and_test_set_inference = False
        if self.number_of_cross_validation_splits > 1:
            raise NotImplementedError("Cross-validation logic is not implemented for this module.")

    def _load_config(self):
        if self.path_augmentation_config is not None:
            self.yaml_config = load_ssl_model_config(self.path_augmentation_config)

    def create_model(self) -> LightningWithInference:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.ssl_type == "simclr":
            model = WrapSimCLRInnerEye(dataset_name=self.dataset_name,
                                       gpus=self.get_num_gpus_to_use(),
                                       encoder_name=self.ssl_encoder.value,
                                       num_samples=self.data_module.num_samples,  # type: ignore
                                       batch_size=self.data_module.batch_size,  # type: ignore
                                       lr=self.learning_rate,
                                       max_epochs=self.num_epochs)
        # Create BYOL model
        else:
            model = WrapBYOLInnerEye(num_samples=self.data_module.num_samples,
                                     learning_rate=self.learning_rate,
                                     dataset_name=self.dataset_name,
                                     encoder_name=self.ssl_encoder.value,
                                     batch_size=self.data_module.batch_size,  # type: ignore
                                     warmup_epochs=10)  # todo (melanibe): this should be config arg
        model.hparams.update({'ssl_type': self.ssl_type})

        self.encoder_output_dim = get_encoder_output_dim(model, self.data_module)

        return model

    def get_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the training and validation steps.
        """
        if hasattr(self, "data_module"):
            return self.data_module

        return self._create_ssl_data_modules(augmentation_config=self.yaml_config,
                                             dataset_name=self.dataset_name,
                                             dataset_path=self.local_dataset,
                                             batch_size=self.batch_size)

    def _create_ssl_data_modules(self,
                                 augmentation_config: ConfigNode,
                                 dataset_name: str,
                                 batch_size: int,
                                 dataset_path: Path) -> LightningDataModule:
        """
        Returns torch lightning data module.
        """
        num_devices = max(1, self.get_num_gpus_to_use())
        if dataset_name == "RSNAKaggle":
            assert dataset_path is not None
            dm = RSNAKaggleDataModule(augmentation_config=augmentation_config,
                                      batch_size=batch_size,
                                      num_workers=self.num_workers,
                                      random_seed=self.random_seed,
                                      use_balanced_binary_loss_for_linear_head=self.use_balanced_binary_loss_for_linear_head,
                                      dataset_path=dataset_path,
                                      num_devices=num_devices)  # type: ignore
        elif dataset_name == "CIFAR10":
            dm = CIFARIEDataModule(num_workers=self.num_workers,
                                   batch_size=batch_size // num_devices,
                                   seed=1234)
            dm.prepare_data()
            dm.setup('fit')
        else:
            raise NotImplementedError(
                f"No pytorch data module implemented for dataset type: {self.dataset_name}")
        return dm

    def get_trainer_arguments(self):
        # Create linear head callback for online monitoring of embedding quality.
        self.online_eval = SSLOnlineEvaluatorInnerEye(class_weights=self.data_module.class_weights,  # type: ignore
                                                      z_dim=self.encoder_output_dim,
                                                      num_classes=self.data_module.num_classes,  # type: ignore
                                                      dataset=self.dataset_name,
                                                      drop_p=0.2)  # type: ignore
        trained_kwargs = {"callbacks": self.online_eval}
        if self.debug:
            trained_kwargs.update({"limit_train_batches": 2, "limit_val_batches": 2})
        return trained_kwargs


'''
class SSLLinearImageClassifierContainer(SSLContainer):
    """
    This container loads a trained SSL encoder and trains a linear head on top of the (frozen) encoder.
    """
    def create_model(self) -> LightningWithInference:
        """
        This method must create the actual Lightning model that will be trained. It can read out parameters from the
        container and pass them into the model, for example.
        """
        # todo need to wrap the logic of checkpoint downloading from a run directly.
        assert self.local_weights_path.exists()
        # Use the local_weight config argument from EssentialParams to pass the SSL checkpoint to
        # LinearImageClassifier
        model = create_ssl_image_classifier(num_classes=self.data_module.num_classes,
                                            pl_checkpoint_path=str(self.local_weights_path))
        # Reset local_weight otherwise the trainer will attempt to recover training from this checkpoint
        # and it will crash instead of just using the loaded weights to fix the encoder.
        self.local_weights_path = None
        return model

    def get_trainer_arguments(self):
        return dict()
'''

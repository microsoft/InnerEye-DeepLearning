#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any

import param
import torch
from pytorch_lightning import LightningDataModule

from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference
from InnerEye.SSL.byol.byol_module import BYOLInnerEye
from InnerEye.SSL.datamodules.utils import create_ssl_data_modules
from InnerEye.SSL.simclr_module import SimCLRInnerEye

from InnerEye.SSL.ssl_classifier_module import SSLOnlineEvaluatorInnerEye, get_encoder_output_dim
from InnerEye.SSL.utils import load_ssl_model_config


class WrapSimCLRInnerEye(SimCLRInnerEye, LightningWithInference):
    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        pass

    def inference_step(self, batch: Any, batch_idx: int, model_output: torch.Tensor):
        pass

    def on_inference_epoch_end(self) -> None:
        pass




class WrapBYOLInnerEye(BYOLInnerEye, LightningWithInference):
    def on_inference_epoch_start(self, dataset_split: ModelExecutionMode, is_ensemble_model: bool) -> None:
        pass

    def inference_step(self, batch: Any, batch_idx: int, model_output: torch.Tensor):
        pass

    def on_inference_epoch_end(self) -> None:
        pass


class SSLContainer(LightningContainer):
    """
    This container is the module to use to train an SSL model either BYOL or SimCLR.
    See Readme for more extensive documentation about its configuration.
    """
    # todo it should be made clear in readme that trainer will use the following fields
    # todo i.e. they have to be set in the init of your module (or defaults will be taken)
    # TODO transform yaml config in nested param class
    # TODO the field output_dir is unused at the moment. Remove it from yaml config

    path_yaml_config = param.String(doc="The path to the yaml config")

    def setup(self):
        self._load_config()
        self.data_module = self.get_data_module()
        self.random_seed = self.yaml_config.train.seed
        self.perform_validation_and_test_set_inference = False
        self.recovery_checkpoint_save_interval = self.yaml_config.train.checkpoint_period
        self.num_epochs = self.yaml_config.scheduler.epochs
        if self.number_of_cross_validation_splits > 1:
            raise NotImplementedError("Cross-validation logic is not implemented for this module.")

    def _load_config(self):
        self.yaml_config = load_ssl_model_config(self.path_yaml_config)

    def create_model(self) -> LightningWithInference:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.yaml_config.train.self_supervision.type == "simclr":
            model = WrapSimCLRInnerEye(output_folder=self.yaml_config.train.output_dir,
                                       dataset_name=self.yaml_config.dataset.name,
                                       gpus=self.get_num_gpus_to_use(),
                                       encoder_name=self.yaml_config.train.self_supervision.encoder_name,
                                       num_samples=self.data_module.num_samples,  # type: ignore
                                       batch_size=self.data_module.batch_size,  # type: ignore
                                       lr=self.yaml_config.train.base_lr)
        # Create BYOL model
        else:
            model = WrapBYOLInnerEye(output_folder=self.yaml_config.train.output_dir,
                                     num_samples=self.data_module.num_samples,
                                     learning_rate=self.yaml_config.train.base_lr,
                                     dataset_name=self.yaml_config.dataset.name,
                                     encoder_name=self.yaml_config.train.self_supervision.encoder_name,
                                     batch_size=self.data_module.batch_size,  # type: ignore
                                     warmup_epochs=10)  # todo (melanibe): this should be config arg
        model.hparams.update({'ssl_type': self.yaml_config.train.self_supervision.type})

        # Create linear head callback for online monitoring of embedding quality.
        self.online_eval = SSLOnlineEvaluatorInnerEye(class_weights=self.data_module.class_weights,  # type: ignore
                                                      z_dim=get_encoder_output_dim(model, self.data_module),
                                                      num_classes=self.data_module.num_classes,  # type: ignore
                                                      dataset=self.yaml_config.dataset.name,
                                                      drop_p=0.2)  # type: ignore
        return model

    def get_data_module(self) -> LightningDataModule:
        """
        Gets the data that is used for the training and validation steps.
        """
        return create_ssl_data_modules(self.yaml_config, self.local_dataset)

    def get_trainer_arguments(self):
        return {"callbacks": self.online_eval}

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

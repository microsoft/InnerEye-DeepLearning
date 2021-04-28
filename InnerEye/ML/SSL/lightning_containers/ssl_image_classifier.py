#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

import param

from InnerEye.ML.SSL.datamodules_and_datasets.datamodules import InnerEyeVisionDataModule
from InnerEye.ML.SSL.lightning_containers.ssl_container import InnerEyeDataModuleTypes, SSLContainer
from InnerEye.ML.SSL.utils import create_ssl_image_classifier
from InnerEye.ML.lightning_container import LightningModuleWithOptimizer


class SSLClassifierContainer(SSLContainer):
    """
    This module is usef to train a linear classifier on top of a frozen (or not) encoder.

    You need to specify:
        todo
    """
    freeze_encoder = param.Boolean(default=True, doc="Whether to freeze the pretrained encoder or not.")
    local_ssl_weights_path = param.ClassSelector(class_=Path, default=None, doc="Local path to SSL weights")

    def create_model(self) -> LightningModuleWithOptimizer:
        """
        This method must create the actual Lightning model that will be trained.
        """
        if self.local_ssl_weights_path is None:
            assert self.extra_downloaded_run_id is not None
            try:
                path_to_checkpoint = self.extra_downloaded_run_id.get_best_checkpoint_paths()
            except FileNotFoundError:
                path_to_checkpoint = self.extra_downloaded_run_id.get_recovery_checkpoint_paths()
            path_to_checkpoint = path_to_checkpoint[0]  # type: ignore
        else:
            path_to_checkpoint = self.local_ssl_weights_path
        assert isinstance(self.data_module, InnerEyeVisionDataModule)
        model = create_ssl_image_classifier(num_classes=self.data_module.dataset_train.dataset.num_classes,
                                            pl_checkpoint_path=str(path_to_checkpoint),
                                            freeze_encoder=self.freeze_encoder,
                                            class_weights=self.data_module.class_weights)

        return model

    def get_data_module(self) -> InnerEyeDataModuleTypes:
        """
        Gets the data that is used for the training and validation steps.
        Here we use different data loader for training of linear head and training of SSL model.
        """
        if hasattr(self, "data_module"):
            return self.data_module
        self.data_module = self._create_ssl_data_modules(linear_head_module=True)
        if self.use_balanced_binary_loss_for_linear_head:
            self.data_module.class_weights = self.data_module.compute_class_weights()
        return self.data_module

    def get_trainer_arguments(self) -> Dict[str, Any]:
        trained_kwargs = {}
        if self.is_debug_model:
            trained_kwargs.update({"limit_train_batches": 2, "limit_val_batches": 2})
        return trained_kwargs

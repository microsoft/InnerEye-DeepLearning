#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.lightning_container import LightningContainer, LightningWithInference


class DummyContainerWithAzureDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(azure_dataset_id="azure_dataset", local_dataset=local_dataset)


class DummyContainerWithoutDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        return LightningWithInference()


class DummyContainerWithLocalDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(local_dataset=local_dataset)


class DummyContainerWithAzureAndLocalDataset(LightningContainer):
    def __init__(self):
        super().__init__()

    def create_lightning_module(self) -> LightningWithInference:
        local_dataset = full_ml_test_data_path("lightning_module_data")
        return LightningWithInference(azure_dataset_id="azure_dataset", local_dataset=local_dataset)

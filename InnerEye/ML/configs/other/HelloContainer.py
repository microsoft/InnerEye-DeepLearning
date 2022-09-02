#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torchmetrics.regression import MeanAbsoluteError
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold

from InnerEye.Common import fixed_paths
from InnerEye.ML.lightning_container import LightningContainer


class HelloDataset(Dataset):
    """
    A simple 1dim regression task, read from a data file stored in the test data folder.
    """

    # Creating the data file:
    # import numpy as np
    # import torch
    #
    # N = 100
    # x = torch.rand((N, 1)) * 10
    # y = 0.2 * x + 0.1 * torch.randn(x.size())
    # xy = torch.cat((x, y), dim=1)
    # np.savetxt("InnerEye/ML/configs/other/hellocontainer.csv", xy.numpy(), delimiter=",")
    def __init__(self, raw_data: List[List[float]]) -> None:
        """
        Creates the 1-dim regression dataset.

        :param raw_data: The raw data, e.g. from a cross validation split or loaded from file. This
            must be numeric data which can be converted into a tensor. See the static method
        from_path_and_indexes for an example call.
        """
        super().__init__()
        self.data = torch.tensor(raw_data, dtype=torch.float)

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        return {'x': self.data[item][0:1], 'y': self.data[item][1:2]}

    @staticmethod
    def from_path_and_indexes(
            root_folder: Path,
            start_index: int,
            end_index: int) -> 'HelloDataset':
        '''
        Static method to instantiate a HelloDataset from the root folder with the start and end indexes.

        :param root_folder: The folder in which the data file lives ("hellocontainer.csv")
        :param start_index: The first row to read.
        :param end_index: The last row to read (exclusive)
        :return: A new instance based on the root folder and the start and end indexes.
        '''
        raw_data = np.loadtxt(root_folder / "hellocontainer.csv", delimiter=",")[start_index:end_index]
        return HelloDataset(raw_data)


class HelloDataModule(LightningDataModule):
    """
    A data module that gives the training, validation and test data for a simple 1-dim regression task.
    If not using cross validation a basic 50% / 20% / 30% split between train, validation, and test data
    is made on the whole dataset.
    For cross validation (if required) we use k-fold cross-validation. The test set remains unchanged
    while the training and validation data cycle through the k-folds of the remaining data.
    """

    def __init__(
            self,
            root_folder: Path,
            number_of_cross_validation_splits: int = 0,
            cross_validation_split_index: int = 0) -> None:
        super().__init__()
        if number_of_cross_validation_splits <= 1:
            # For 0 or 1 splits just use the default values on the whole data-set.
            self.train = HelloDataset.from_path_and_indexes(root_folder, start_index=0, end_index=50)
            self.val = HelloDataset.from_path_and_indexes(root_folder, start_index=50, end_index=70)
            self.test = HelloDataset.from_path_and_indexes(root_folder, start_index=70, end_index=100)
        else:
            # Raise exceptions for unreasonable values
            if cross_validation_split_index >= number_of_cross_validation_splits:
                raise IndexError(f"The cross_validation_split_index ({cross_validation_split_index}) is too large "
                f"given the number_of_cross_validation_splits ({number_of_cross_validation_splits}) requested")
            raw_data = np.loadtxt(root_folder / "hellocontainer.csv", delimiter=",")
            np.random.seed(42)
            np.random.shuffle(raw_data)
            if number_of_cross_validation_splits >= len(raw_data):
                raise ValueError(f"Asked for {number_of_cross_validation_splits} cross validation splits from a "
                f"dataset of length {len(raw_data)}")
            # Hold out the last 30% as test data
            self.test = HelloDataset(raw_data[70:100])
            # Create k-folds from the remaining 70% of the data-set. Use one for the validation
            # data and the rest for the training data
            raw_data_remaining = raw_data[0:70]
            k_fold = KFold(n_splits=number_of_cross_validation_splits)
            train_indexes, val_indexes = list(k_fold.split(raw_data_remaining))[cross_validation_split_index]
            self.train = HelloDataset(raw_data_remaining[train_indexes])
            self.val = HelloDataset(raw_data_remaining[val_indexes])

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train, batch_size=5)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val, batch_size=5)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test, batch_size=5)


class HelloRegression(LightningModule):
    """
    A simple 1-dim regression model.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(in_features=1, out_features=1, bias=True)
        self.test_mse: List[torch.Tensor] = []
        self.test_mae = MeanAbsoluteError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It runs a forward pass of a tensor through the model.

        :param x: The input tensor(s)
        :return: The model output.
        """
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], *args: Any, **kwargs: Any) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of training data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of training data
        :return: The loss value with a computation graph attached.
        """
        loss = self.shared_step(batch)
        self.log("loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], *args: Any,  # type: ignore
                        **kwargs: Any) -> torch.Tensor:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It consumes a minibatch of validation data (coming out of the data loader), does forward propagation, and
        computes the loss.

        :param batch: The batch of validation data
        :return: The loss value on the validation data.
        """
        loss = self.shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        return loss

    def shared_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        This is a convenience method to reduce code duplication, because training, validation, and test step share
        large amounts of code.

        :param batch: The batch of data to process, with input data and targets.
        :return: The MSE loss that the model achieved on this batch.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        return torch.nn.functional.mse_loss(prediction, target)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It returns the PyTorch optimizer(s) and learning rate scheduler(s) that should be used for training.
        """
        optimizer = Adam(self.parameters(), lr=1e-1)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]

    def on_test_epoch_start(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can prepare data structures that need to be in place before evaluating the model on the
        test set (that is done in the test_step).
        """
        self.test_mse = []
        self.test_mae.reset()

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:  # type: ignore
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        It evaluates the model in "inference mode" on data coming from the test set. It could, for example,
        also write each model prediction to disk.

        :param batch: The batch of test data.
        :param batch_idx: The index (0, 1, ...) of the batch when the data loader is enumerated.
        :return: The loss on the test data.
        """
        input = batch["x"]
        target = batch["y"]
        prediction = self.forward(input)
        # This illustrates two ways of computing metrics: Using standard torch
        loss = torch.nn.functional.mse_loss(prediction, target)
        self.test_mse.append(loss)
        # Metrics computed using PyTorch Lightning objects. Note that these will, by default, attempt
        # to synchronize across GPUs.
        self.test_mae.update(preds=prediction, target=target)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        This method is part of the standard PyTorch Lightning interface. For an introduction, please see
        https://pytorch-lightning.readthedocs.io/en/stable/starter/converting.html
        In this method, you can finish off anything to do with evaluating the model on the test set,
        for example writing aggregate metrics to disk.
        """
        average_mse = torch.mean(torch.stack(self.test_mse))
        Path("test_mse.txt").write_text(str(average_mse.item()))
        Path("test_mae.txt").write_text(str(self.test_mae.compute().item()))


class HelloContainer(LightningContainer):
    """
    An example for using the InnerEye functionality to "bring your own lightning model". This container has methods
    to generate the actual Lightning model, and read out the datamodule that will be used for training.
    The number of training epochs is controlled at container level.
    You can train this model by running `python InnerEye/ML/runner.py --model=HelloContainer` on the local box,
    or via `python InnerEye/ML/runner.py --model=HelloContainer --azureml=True` in AzureML
    """

    def __init__(self) -> None:
        super().__init__()
        self.local_dataset = fixed_paths.repository_root_directory() / "InnerEye" / "ML" / "configs" / "other"
        self.num_epochs = 20

    # This method must be overridden by any subclass of LightningContainer. It returns the model that you wish to
    # train, as a LightningModule
    def create_model(self) -> LightningModule:
        return HelloRegression()

    # This method must be overridden by any subclass of LightningContainer. It returns a data module, which
    # in turn contains 3 data loaders for training, validation, and test set.
    #
    # If the container is used for cross validation then this method must handle the cross validation splits.
    # Because this deals with data loaders, not loaded data, we cannot check automatically that cross validation is
    # handled correctly within the LightningContainer base class, i.e. if you forget to do the cross validation split
    # in your subclass nothing will fail, but each child run will be identical since they will each be given the full
    # dataset.
    def get_data_module(self) -> LightningDataModule:
        assert self.local_dataset is not None
        return HelloDataModule(
            root_folder=self.local_dataset,
            number_of_cross_validation_splits=self.number_of_cross_validation_splits,
            cross_validation_split_index=self.cross_validation_split_index)  # type: ignore

    # This is an optional override: This report creation method can read out any files that were written during
    # training, and cook them into a nice looking report. Here, the report is a simple text file.
    def create_report(self) -> None:
        # This just prints out the test MSE, but you could also generate a Jupyter notebook here, for example.
        test_mse = Path("test_mse.txt").read_text().strip()
        test_mae = Path("test_mae.txt").read_text().strip()
        report = f"Performance on test set: MSE = {test_mse}, MAE = {test_mae}"
        print(report)
        Path("report.txt").write_text(report)

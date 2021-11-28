"""
A minimum working example of a WSI tile classifier using PyTorch Lightening modules
Includes training, validation, and testing
Run python InnerEyePrivate/ML/runner.py --model=PandaTileClassificationV0
Tensorboard logging of training and validation viewed by running tensorboard --logdir ~/outputs/<output directory>
Test loss and accuracy are displayed at the end of training and testing
"""

from pathlib import Path
from typing import Any, Dict

import torch
import torch.optim
import torchvision.models
from monai.data.dataset import Dataset
from pytorch_lightning import LightningDataModule, LightningModule
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms import Normalize

from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.utils.split_dataset import DatasetSplits
from health_ml.data.histopathology.datasets.default_paths import PANDA_TILES_DATASET_DIR
from health_ml.data.histopathology.datasets.panda_tiles_dataset import PandaTilesDataset
from InnerEyePrivate.Histopathology.models.transforms import LoadTiled


class PandaTileClassifier(LightningModule):
    def __init__(self, label_column: str, n_classes: int, tile_size: int) -> None:
        super().__init__()

        # Define the dataset specific attributes
        self.label_column = label_column
        self.n_classes = n_classes
        self.tile_size = tile_size
        self.save_hyperparameters()

        self.preprocessing_fn = Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        # Define PyTorch model
        self.model = torchvision.models.resnet18(pretrained=True)
        # replace number of output classes in the last layer
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features,
                                        out_features=self.n_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.activation_fn = nn.Softmax()

        # Metrics Objects
        self.train_metrics = self.get_metrics()
        self.val_metrics = self.get_metrics()
        self.test_metrics = self.get_metrics()

    def forward(self, images: torch.Tensor) -> Any:  # type:ignore
        return self.model(images)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters())

    def get_metrics(self) -> nn.ModuleDict:
        return nn.ModuleDict({'accuracy': Accuracy(num_classes=self.n_classes, average='micro'),
                              'macro_accuracy': Accuracy(num_classes=self.n_classes, average='macro'),
                              'weighted_accuracy': Accuracy(num_classes=self.n_classes, average='weighted')})

    def log_metrics(self,
                    stage: str) -> None:
        valid_stages = ['train', 'test', 'val']
        if stage not in valid_stages:
            raise Exception(f"Invalid stage. Chose one of {valid_stages}")
        for metric_name, metric_object in getattr(self, f'{stage}_metrics').items():
            self.log(f'{stage}/{metric_name}', metric_object, on_epoch=True, on_step=False, logger=True, sync_dist=True)

    def _shared_step(self, batch: Dict, batch_idx: int, stage: str) -> Dict[str, torch.Tensor]:
        images = self.preprocessing_fn(batch['image'])
        labels = batch[self.label_column]
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        probs = self.activation_fn(logits)
        preds = torch.argmax(probs, dim=1)
        results = dict()
        for metric_object in getattr(self, f'{stage}_metrics').values():
            metric_object.update(preds.view(-1, 1), labels.view(-1, 1))
        # TODO: Properly log tile attention scores
        results.update({'loss': loss, 'pred_label': preds, 'true_label': labels})
        return results

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:  # type: ignore
        train_result = self._shared_step(batch, batch_idx, 'train')
        self.log('train/loss', train_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        self.log_metrics('train')
        return train_result['loss']

    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:  # type: ignore
        val_result = self._shared_step(batch, batch_idx, 'val')
        self.log('val/loss', val_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        self.log_metrics('val')
        return val_result['loss']

    def test_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:  # type: ignore
        fields_to_copy = ['slide_id', 'tile_id', 'tile_x', 'tile_y']
        test_result = self._shared_step(batch, batch_idx, 'test')
        self.log('test/loss', test_result['loss'], on_epoch=True, on_step=True, logger=True, sync_dist=True)
        self.log_metrics('test')
        test_result.update({field: batch[field] for field in fields_to_copy})
        return test_result['loss']


class PandaTilesDataModule(LightningDataModule):
    def __init__(self, root_path: Path):
        super().__init__()
        self.root_path = root_path
        self.batch_size = 256
        self.preprocessing = LoadTiled('image')
        self.dataset = PandaTilesDataset(self.root_path)
        self.splits = DatasetSplits.from_proportions(self.dataset.dataset_df.reset_index(), .8, .1, .1,
                                                     subject_column='tile_id',
                                                     group_column='slide_id')  # splitting by slides
        # as PANDA doesn't have patient ID
        self.train_dataset = PandaTilesDataset(self.root_path, dataset_df=self.splits.train)
        self.val_dataset = PandaTilesDataset(self.root_path, dataset_df=self.splits.val)
        self.test_dataset = PandaTilesDataset(self.root_path, dataset_df=self.splits.test)

    def _get_dataloader(self, dataset: PandaTilesDataset) -> DataLoader:
        transformed_dataset = Dataset(dataset, transform=self.preprocessing)  # type: ignore
        return DataLoader(transformed_dataset, batch_size=self.batch_size, num_workers=5)

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset)


class PandaTileClassificationV0(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        self.local_dataset = Path(PANDA_TILES_DATASET_DIR)
        self.num_epochs = 100

    def create_model(self) -> PandaTileClassifier:
        return PandaTileClassifier(label_column=PandaTilesDataset.LABEL_COLUMN,
                                   n_classes=PandaTilesDataset.N_CLASSES, tile_size=224)

    def get_data_module(self) -> PandaTilesDataModule:
        return PandaTilesDataModule(root_path=self.local_dataset)


if __name__ == '__main__':

    container = PandaTileClassificationV0()
    data_module = container.get_data_module()
    print(data_module.dataset[0])
    train_data_loader = data_module.train_dataloader()

    module = container.create_model()
    print(module.model)

    for batch_idx, batch in enumerate(train_data_loader):
        print(module.training_step(batch, batch_idx))
        break

    test_data_loader = data_module.test_dataloader()
    for batch_idx, batch in enumerate(test_data_loader):
        print(module.test_step(batch, batch_idx))
        break

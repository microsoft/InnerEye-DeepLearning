from typing import Any, Optional, Dict, List, Union

from pl_bolts.datamodules.vision_datamodule import VisionDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader

from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader

from InnerEye.SSL.datamodules.transforms_utils import InnerEyeCIFAREvalTransform, InnerEyeCIFARTrainTransform
from InnerEye.SSL.utils import SSLModule

DATASET_CLS = {SSLModule.ENCODER: CIFAR10, SSLModule.LINEAR_HEAD: CIFAR10}
TRAIN_TRANSFORMS = InnerEyeCIFARTrainTransform(32)
VAL_TRANSFORMS = InnerEyeCIFAREvalTransform(32)


class CIFARIEDataModule(VisionDataModule):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        """
        super().__init__(**kwargs)
        self.class_weights = None
        self.drop_last = True

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        """
        Saves files to data_dir
        """
        for _cls in DATASET_CLS.values():
            _cls(self.data_dir, train=True, download=True)
            _cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Creates train, val, and test dataset
        """
        self.datasets: Dict[SSLModule, List] = {name: list() for name in DATASET_CLS.keys()}
        for _cls_name, _cls in DATASET_CLS.items():
            dataset_train = _cls(self.data_dir, True, transform=TRAIN_TRANSFORMS)
            dataset_val = _cls(self.data_dir, True, transform=VAL_TRANSFORMS)

            # Split training and validation sets
            self.datasets[_cls_name].append(self._split_dataset(dataset_train))
            self.datasets[_cls_name].append(self._split_dataset(dataset_val, train=False))

        if stage == "test":
            raise NotImplementedError

    def train_dataloader(self, *args: Any, **kwargs: Any) -> Dict[SSLModule, DataLoader]:
        """
        The train dataloaders
        """
        dataloaders = {
            SSLModule.ENCODER: self._data_loader(self.datasets[SSLModule.ENCODER][0], shuffle=self.shuffle),
            SSLModule.LINEAR_HEAD: self._data_loader(self.datasets[SSLModule.LINEAR_HEAD][0], shuffle=self.shuffle)}

        return dataloaders

    def val_dataloader(self, *args: Any, **kwargs: Any) -> CombinedLoader:
        """ The val dataloader
        """
        dataloaders = {
            SSLModule.ENCODER: self._data_loader(self.datasets[SSLModule.ENCODER][1]),
            SSLModule.LINEAR_HEAD: self._data_loader(self.datasets[SSLModule.LINEAR_HEAD][1])}

        return CombinedLoader(dataloaders, mode="max_size_cycle")

    @property
    def num_samples(self) -> int:
        return len(self.datasets[SSLModule.ENCODER][0])

    @property
    def num_classes(self) -> int:
        return len(set(self.datasets[SSLModule.LINEAR_HEAD][0].dataset.targets))

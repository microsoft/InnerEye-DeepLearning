#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
from numpy.core.numeric import full
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
import torch
import pickle
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import torchio as tio


class Prostate2DSimpleDataset(Dataset):
    def __init__(self, base_path, csv_name, _set: str, input_channels: int = 3, 
                 labels=None, k_shots=None, augment=False, transforms=None, transforms_args=None):
        self.base_path = base_path
        self.df = pd.read_csv(os.path.join(base_path, csv_name))
        if _set is not None:
            self.df = self.df.loc[self.df.set == _set]
        if k_shots:
            idx = self.df.index
            idx = idx[:k_shots]
            self.df = self.df.loc[idx]
        self.input_channels = input_channels
        self.labels = labels
        self.augment = augment
        if augment:
            self.log_gamma = (-.5, .5)
            self.blur_std = .5
            self.noise_std = .125
        self.transforms = transforms
        self.transforms_args = transforms_args

    def __len__(self):
        return len(self.df)

    def augmentations(self):
        i_augs = []
        l_augs = []
        if np.random.rand() >= .5:
            i_augs.append(tio.Flip(2))
            l_augs.append(tio.Flip(2))
        gamma = np.exp(np.random.uniform(*self.log_gamma))
        blur_std = np.random.uniform(0, self.blur_std)
        noise_std = np.random.uniform(0, self.noise_std)
        noise_seed = np.random.randint(0, 255)
        i_augs += [tio.Gamma(gamma), 
                   tio.Blur(blur_std),
                   tio.Noise(0, noise_std, noise_seed)]
        # l_augs.append(tio.Blur(blur_std))
        img_aug = tio.Compose(i_augs)
        seg_aug = tio.Compose(l_augs)
        return img_aug, seg_aug

    def __getitem__(self, i):
        i = self.df.index[i]
        path = os.path.join(self.base_path, self.df.loc[i, 'file'])
        with open(path, 'rb') as f:
            img = pickle.load(f)
        # replicate the image to have 'input_channels' num of channels
        img = np.stack([img] * self.input_channels, 0)

        # load labels
        if self.labels:
            img_label = [np.zeros((128, 128))]
            for label in self.labels:
                if label == 'femurs':
                    path = os.path.join(self.base_path, self.df.loc[i, 'femur_l'])
                    with open(path, 'rb') as f:
                        img_label.append(pickle.load(f))
                    path = os.path.join(self.base_path, self.df.loc[i, 'femur_r'])
                    with open(path, 'rb') as f:
                        img_label[-1] += pickle.load(f)
                else:
                    path = os.path.join(self.base_path, self.df.loc[i, label])
                    with open(path, 'rb') as f:
                        img_label.append(pickle.load(f))
            img_label = np.stack(img_label, 0)
            img_label[0] = (img_label[1:].sum(0) == 0).astype(int)

            if self.augment:
                img_aug, seg_aug = self.augmentations()
                img = img_aug(img[:,:,:, np.newaxis]).squeeze(-1)
                img_label = seg_aug(img_label[:,:,:, np.newaxis]).squeeze(-1)
            return torch.Tensor(img).float(), torch.Tensor(img_label).int()

        else:
            if self.augment:
                raise NotImplementedError
            if self.transforms is not None:
                if self.transforms_args is not None:
                    img = self.transforms(img, **self.transforms_args)
                else: 
                    img = self.transforms(img)
            return torch.Tensor(img).float()

class Prostate2DSimpleDataLoader(LightningDataModule):
    """
    A data module that gives the training, validation and test data for a simple 1-dim regression task.
    """

    def __init__(self,
                 dataset_path: str,
                 csv_base_name: str,
                 batch_size: int = 32,
                 input_channels: int = 3,
                 num_workers: int = 12,
                 labels: list = None,
                 k_shots: int = None,
                 augment: bool = False,
                 **kargs
                 ) -> None:
        super().__init__()
        print('dataset path: {}'.format(os.path.join(dataset_path, csv_base_name)), flush=True)
        assert os.path.isfile(os.path.join(dataset_path, csv_base_name)), dataset_path + '/' + csv_base_name + ' should be a csv file'
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train = Prostate2DSimpleDataset(dataset_path, csv_base_name, 'train', 
                                             input_channels, labels=labels, k_shots=k_shots, augment=augment)
        self.val = Prostate2DSimpleDataset(dataset_path, csv_base_name, 'val', 
                                           input_channels, labels=labels)
        self.test = Prostate2DSimpleDataset(dataset_path, csv_base_name, 'test', 
                                            input_channels, labels=labels)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

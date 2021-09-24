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
from loaders.prostate_loader import Prostate2DSimpleDataset


class Prostate3D3SlicesSimpleDataset(Prostate2DSimpleDataset):
    def __init__(self, base_path, csv_name, _set: str, input_channels: int = 3, 
                 labels=None, k_shots=None, augment=False, transforms=None, transforms_args=None):
        # todo: check k shots here makes sense 
        if k_shots:
            raise NotImplementedError
        super().__init__(base_path=base_path, csv_name=csv_name, _set=_set, input_channels=input_channels,
                         labels=labels, k_shots=k_shots, augment=augment, transforms=transforms, transforms_args=transforms_args)
        # reordering the list of patients in order of scan slice so that we can query three consecutive slices
        n_slice = [int(f.rsplit('ct_slice_')[1].rsplit('.pkl')[0]) for f in self.df.file]
        self.df['n_slice'] = n_slice
        self.df = self.df.sort_values(by=['subject', 'n_slice'])
        self.df = self.df.reset_index().drop(columns='index')
        
        self.idx_to_query = list(self.df.loc[(self.df.n_slice !=0) & (self.df.n_slice !=23)].index)
 
    def __len__(self):
        return len(self.idx_to_query)

    def __getitem__(self, i):
        i = self.idx_to_query[i]
        out = [super(Prostate3D3SlicesSimpleDataset, self).__getitem__(_i) for _i in np.arange(i-1, i+2, 1)]
        if len(out[0]) == 2:
            return torch.stack([o[0][0] for o in out], 0), torch.stack([o[1] for o in out])
        else:
            return torch.stack([o[0] for o in out], 0).float()
 

class Prostate3D3SlicesSimpleDataLoader(LightningDataModule):
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

        self.train = Prostate3D3SlicesSimpleDataset(dataset_path, csv_base_name, 'train', 
                                             input_channels, labels=labels, k_shots=k_shots, augment=augment)
        self.val = Prostate3D3SlicesSimpleDataset(dataset_path, csv_base_name, 'val', 
                                           input_channels, labels=labels)
        self.test = Prostate3D3SlicesSimpleDataset(dataset_path, csv_base_name, 'test', 
                                            input_channels, labels=labels)

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import pickle
from torch.utils.data.dataset import Dataset
import torch

class SynthDataset(Dataset):
    def __init__(self, path, _set, debug=False):
        assert os.path.isdir(str(path)), 'path not recognised'
        if debug:
            if _set == 'train':
                self.files = [os.path.join(str(path), '{}.pkl'.format(i)) for i in range(560)]
            else:
                self.files = [os.path.join(str(path), '{}.pkl'.format(i)) for i in range(560, 630)]
        else:
            if _set == 'train':
                self.files = [os.path.join(str(path), '{}.pkl'.format(i)) for i in range(9000)]
            else:
                self.files = [os.path.join(str(path), '{}.pkl'.format(i)) for i in range(9000, 10000)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            data = pickle.load(f)
        x = data[0].unsqueeze(0).float()
        y = torch.stack([data[1]==i for i in range(4)],0).long()
        return x, y

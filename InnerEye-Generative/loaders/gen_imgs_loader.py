#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import os
import numpy as np
import torch
import pickle
from torch.utils.data.dataset import Dataset
import SimpleITK as sitk
from tqdm import tqdm


class LatentRepsDS(Dataset):
    def __init__(self, 
                 dataset_path, 
                 n_scans, 
                 samples, 
                 preload=True, 
                 shuffle=True,
                 background_perc=1,
                 add_scan_to_latent=False) -> None:
        if isinstance(n_scans, int):
            self.n_scans = np.arange(n_scans)
        elif isinstance(n_scans, list):
            self.n_scans = np.array(n_scans)
        assert (background_perc >= 0) & (background_perc <= 1), \
            'background_perc should be between 0 and 1'
        self.background_perc = background_perc
        max_samples = 128 * 128
        if samples is None:
            samples = max_samples
        self.samples = min((samples, max_samples))
        self.dataset_path = dataset_path
        print('Note: we assume the following labels in the following order: femurs, bladder, prostate')
        self.shuffle = shuffle
        self.preload = preload
        self.add_scan_to_latent = add_scan_to_latent
        self.x = None
        self.y = None
        self.pixel_idx = None
        self.scan_idx = None
        if preload:
            # todo: make in list
            for j, i in tqdm(enumerate(self.n_scans)):
                _x, _y, _ = self.prep_slice(i)
                if j == 0:
                    x = np.empty((0, _x.shape[1]))
                    y = np.empty((0))

                x = np.concatenate((x, _x), 0)
                y = np.concatenate((y, _y), 0)
            self.x = x
            self.y = y
            self.len = self.x.shape[0]
        else:
            pixel_idx = np.empty(0).astype(int)
            scan_idx = np.empty(0).astype(int)
            for i in tqdm(self.n_scans):
                _, _, _idx = self.prep_slice(i)
                pixel_idx = np.concatenate((pixel_idx, _idx), 0)
                scan_idx = np.concatenate((scan_idx, np.ones_like(_idx) * i), 0)
            self.pixel_idx = pixel_idx
            self.scan_idx = scan_idx
            self.len = self.scan_idx.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.preload:
            return torch.Tensor(self.x[idx]).float(), torch.Tensor(self.y[idx, np.newaxis]).long()
        else:
            x, y = self.load_items(self.scan_idx[idx])
            return torch.Tensor(x[self.pixel_idx[idx]]).float(),\
                torch.Tensor(y[self.pixel_idx[idx], np.newaxis]).long()

    def prep_slice(self, i):
        z, y = self.load_items(i)

        # calc label prevalence
        perc = np.array([np.mean(y == i) for i in range(4)])
        # recalibrate the percentage of background items
        perc[0] = perc[0] * self.background_perc
        if perc[1:].sum() > 0:
            perc[1:] = perc[1:] * (1 - perc[0]) / perc[1:].sum()
        samples = (perc * self.samples).astype(int)
        samples[(perc > 0) & (samples == 0)] = 1

        # generate the idx we want to sample
        idx = np.empty(0).astype(int)
        for i in range(4):
            pos_idx = np.where(y == i)[0]
            if self.shuffle:
                np.random.shuffle(pos_idx)
            idx = np.concatenate((idx, pos_idx[: samples[i]]))

        return z[idx], y[idx], idx

    def load_latent(self, i):
        # load latent vec
        file_path = os.path.join(self.dataset_path, '{}.pkl'.format(i))
        with open(file_path, 'rb') as f:
            z = pickle.load(f)
        z = z.reshape((-1, z.shape[0]))
        return z

    def load_scan(self, i):
        # load scan
        file_path = os.path.join(self.dataset_path, 'scans', '{}.nii.gz'.format(i))
        if not os.path.isfile(file_path):
            raise FileNotFoundError('cannot find patient {}'.format(i))
        scan = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        scan = np.flip(scan, 1)
        a = 2. / (155. - (-100.))
        b = 1. - a * 155.
        scan = scan * a + b
        return scan.flatten()

    def load_labels(self, i):
        # load labels
        y = np.zeros((1, 128, 128))
        for label in ['femur_l', 'femur_r']:
            file_path = os.path.join(self.dataset_path,  'seg', '{}_{}.nii.gz'.format(i, label))
            if os.path.isfile(file_path):
                map = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
                map = np.flip(map, 1)
                y += map
        for j, label in enumerate(['bladder', 'prostate']):
            file_path = os.path.join(self.dataset_path, 'seg', '{}_{}.nii.gz'.format(i, label))
            if os.path.isfile(file_path):
                map = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
                map = np.flip(map, 1)
                y += map * (j + 2)

        # make sure there's no overlap between prostate and bladder
        # bladder is class 2 and prostate is class 3, 
        # hence want to make sure there's no item in class 5
        y[y == 5] = 3

        # add background class
        y = y.flatten()
        return y

    def load_items(self, i):
        items = []
        items.append(self.load_latent(i))
        if self.add_scan_to_latent:
            items[-1] = np.concatenate((items[-1], self.load_scan(i).reshape((-1, 1))), -1)
        items.append(self.load_labels(i))
        return items


class ValidationLatentRepsDS(LatentRepsDS):
    def __init__(self, 
                 dataset_path, 
                 n_scans,
                 add_scan_to_latent=False) -> None:
        self.dataset_path = dataset_path
        self.add_scan_to_latent = add_scan_to_latent
        if isinstance(n_scans, int):
            self.n_scans = np.arange(n_scans)
        elif isinstance(n_scans, list):
            self.n_scans = np.array(n_scans)

        for j, i in tqdm(enumerate(self.n_scans)):
            _x, _y, _s = self.load_items(i)
            if j == 0:
                x = np.empty((0, _x.shape[1]))
                y = np.empty((0))
                s = np.empty((0))

            x = np.concatenate((x, _x), 0)
            y = np.concatenate((y, _y), 0)
            s = np.concatenate((s, _s), 0)
        self.x = x
        self.y = y
        self.s = s
        self.len = len(self.x)

    def __getitem__(self, idx):
        # print('idx {} --'.format(idx), flush=True)
        return torch.Tensor(self.x[idx]).float(), \
            torch.Tensor(self.y[idx, np.newaxis]).long(), \
            torch.Tensor(self.s[idx, np.newaxis]).float(), \
            torch.Tensor([idx]).int()


    def load_items(self, i):
        items = super().load_items(i)
        items.append(self.load_scan(i))
        return items




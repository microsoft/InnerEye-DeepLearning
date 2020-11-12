#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Tuple

import h5py
import pandas as pd
import torch

from InnerEye.ML.dataset.full_image_dataset import GeneralDataset
from InnerEye.ML.reconstruction_config import ReconstructionModelBase

KSPACE_NAME = r'kspace'
RECONSTRUCTION_NAME = r'reconstruction_rss'
COIL_SENSIVITY_NAME = r'coil_sensitivities'
FILE_PREFIX_NAME = r'dataset'


class FastMriDataset(GeneralDataset):
    """
    FastMRI challenge dataset
    """

    def __init__(self, args: ReconstructionModelBase, data_frame: pd.DataFrame):
        super().__init__(args, data_frame)

        # Check base_path
        assert self.args.local_dataset is not None
        if not self.args.local_dataset.is_dir():
            raise ValueError("local_dataset should be the path to the base directory of the data: {}".
                             format(self.args.local_dataset))

    def __len__(self) -> int:
        return len(self.data_frame.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data_frame.loc[idx]
        
        f = item['FilePath']
        s = item['SliceIndex']

        with h5py.File(f) as d:
            kspace = torch.tensor(d[KSPACE_NAME][s, :])
            recon = torch.tensor(d[RECONSTRUCTION_NAME][s, :])
            return (kspace, recon)
  

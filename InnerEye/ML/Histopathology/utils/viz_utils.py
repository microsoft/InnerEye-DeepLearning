#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import math
from typing import Any, Dict

import matplotlib.pyplot as plt
from monai.data.dataset import Dataset
from monai.data.image_reader import WSIReader
from torch.utils.data import DataLoader

from InnerEye.ML.Histopathology.datasets.panda_dataset import PandaDataset, LoadPandaROId
from InnerEye.ML.Histopathology.utils.naming import SlideKey


def load_image_dict(sample: dict, level: int, margin: int) -> Dict[SlideKey, Any]:
    """
    Load image from metadata dictionary
    :param sample: dict describing image metadata. Example:
        {'image_id': ['1ca999adbbc948e69783686e5b5414e4'],
        'image': ['/tmp/datasets/PANDA/train_images/1ca999adbbc948e69783686e5b5414e4.tiff'],
         'mask': ['/tmp/datasets/PANDA/train_label_masks/1ca999adbbc948e69783686e5b5414e4_mask.tiff'],
         'data_provider': ['karolinska'],
         'isup_grade': tensor([0]),
         'gleason_score': ['0+0']}
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :return: a dict containing the image data and metadata
    """
    loader = LoadPandaROId(WSIReader('cuCIM'), level=level, margin=margin)
    img = loader(sample)
    return img


def plot_panda_data_sample(panda_dir: str, nsamples: int, ncols: int, level: int, margin: int,
                           title_key: str = 'data_provider') -> None:
    """
    :param panda_dir: path to the dataset, it's expected a file called "train.csv" exists at the path.
        Look at the PandaDataset for more detail
    :param nsamples: number of random samples to be visualized
    :param ncols: number of columns in the figure grid. Nrows is automatically inferred
    :param level: level of resolution to be loaded
    :param margin: margin to be included
    :param title_key: metadata key in image_dict used to label each subplot
    """
    panda_dataset = Dataset(PandaDataset(root=panda_dir))[:nsamples]  # type: ignore
    loader = DataLoader(panda_dataset, batch_size=1)

    nrows = math.ceil(nsamples/ncols)
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(9, 9))

    for dict_images, ax in zip(loader, axes.flat):
        slide_id = dict_images[SlideKey.SLIDE_ID]
        title = dict_images[SlideKey.METADATA][title_key]
        print(f">>> Slide {slide_id}")
        img = load_image_dict(dict_images, level=level, margin=margin)
        ax.imshow(img[SlideKey.IMAGE].transpose(1, 2, 0))
        ax.set_title(title)
    fig.tight_layout()

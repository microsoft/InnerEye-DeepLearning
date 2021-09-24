#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import SimpleITK as sitk
import os 
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from locations import GENSCANSPATH
GENSCANSPATH = Path(GENSCANSPATH)


def plot_scans(path, fig_name, _labels=True):
    colours = np.array([[0.,0.,1.], [0.13,0.4,0.], [1., .1, .1]])
    colours = colours[:, :, np.newaxis, np.newaxis]
    a = 2. / (155. - (-100.))
    b = 1. - a * 155
    fig, axs = plt.subplots(2, 8, figsize=(28, 7))
    for i, ax in enumerate(axs.flatten()):
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, 'scans', '{}.nii.gz'.format(i))))[0]
        img = np.flip(img, 0)
        img = img * a + b
        img = np.stack([img] * 3, 0)
        img = (img + 1.) / 2.
        if _labels:
            labels = np.empty((0, 128, 128))
            for tag in ['femur_l', 'femur_r', 'bladder', 'prostate']:
                file = os.path.join(path, 'seg', '{}_{}.nii.gz'.format(i, tag))
                if os.path.isfile(file):
                    label = sitk.GetArrayFromImage(sitk.ReadImage(file))
                else:
                    label = np.zeros((1, 128, 128))
                labels = np.concatenate((labels, label), 0)
            labels = np.flip(labels, 1)
            labels[1] = labels[0] + labels[1]
            labels[2, labels[3] == 1] = 0
            labels = labels[1:]
            labels = np.stack([labels] * 3, 1)
            clabel = (labels * colours).sum(0)
            img_n_label = np.moveaxis(img * .5 + clabel, 0, -1)
            ax.imshow(img_n_label)
        else:
            img = np.moveaxis(img, 0, -1)
            ax.imshow(img)
        ax.set_title(i)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.show()


if __name__ == '__main__':

    path = GENSCANSPATH / 'train'
    plot_scans(path, GENSCANSPATH / 'train.png', False)
    plot_scans(path, GENSCANSPATH / 'train_with_labels.png', True)
    path = GENSCANSPATH / 'val'
    plot_scans(path, GENSCANSPATH / 'val.png', False)
    plot_scans(path, GENSCANSPATH / 'val_with_labels.png', True)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from tqdm.notebook import tqdm
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import pickle
import sys
import os
import numpy as np
from argparse import ArgumentParser
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from loaders.transformation_utils import resample_image
from locations import DATASETPATH
DATASETPATH = Path(DATASETPATH)


def main():
    os.makedirs(DATASETPATH / '2D', exist_ok=True)
    os.makedirs(DATASETPATH / '2D' / 'scans', exist_ok=True)
    os.makedirs(DATASETPATH / '2D' / 'labels', exist_ok=True)

    df = pd.read_csv(DATASETPATH / 'dataset.csv')
    res = 4
    for pat in tqdm(df.subject.unique()):
        for el in df.channel.unique():
            img = sitk.ReadImage(DATASETPATH / df.loc[(df.subject == pat) & (df.channel == el), 'filePath'].item())
            img = resample_image(img, (res, res, res), is_label=True if el != 'ct' else False)
            img = sitk.GetArrayFromImage(img)
            # find start / end points
            s = [max(int(sh/2)-w, 0) for sh, w in zip(img.shape, [24, 64, 64])]
            e = [min(int(sh/2)+w, sh) for sh, w in zip(img.shape, [24, 64, 64])]
            img = img[
                s[0]: e[0]: 2,
                s[1]: e[1],
                s[2]: e[2]
                    ]
            if el == 'ct':
                img = np.clip(img, -100, 155)
                a = 2. / (155. - (-100.))
                b = 1 - a * 155
                img = img * a + b
                img_temp = np.ones((24, 128, 128)) * -1
                s = [int((fi-i)/2) for fi, i in zip(img_temp.shape, img.shape)]
                img_temp[
                    s[0]: s[0] + img.shape[0],
                    s[1]: s[1] + img.shape[1],
                    s[2]: s[2] + img.shape[2]
                        ] = img.copy()
                img = img_temp.copy()
                for slice in range(img.shape[0]):
                    with open(DATASETPATH / '2D' / 'scans' / 'img_pat_{}_{}_slice_{}.pkl'.format(pat, el, slice), 'wb') as f:
                        pickle.dump(img[slice], f)
            else:
                img_temp = np.zeros((24, 128, 128)) 
                s = [int((fi-i)/2) for fi, i in zip(img_temp.shape, img.shape)]
                img_temp[
                    s[0]: s[0] + img.shape[0],
                    s[1]: s[1] + img.shape[1],
                    s[2]: s[2] + img.shape[2]
                        ] = img.copy()
                img = img_temp.copy()
                for slice in range(img.shape[0]):
                    with open(DATASETPATH / '2D' / 'labels' / 'img_pat_{}_{}_slice_{}.pkl'.format(pat, el, slice), 'wb') as f:
                        pickle.dump(img[slice], f)

    # generate dataset file
    labels = ['file', 'femur_l', 'femur_r', 'rectum', 'prostate', 'seminalvesicles']
    elements = ['ct'] + labels[1:]
    Dict = {'subject': []}
    for l in labels:
        Dict[l] = []

    for pat in tqdm(df.subject.unique()):
        for slice in range(24):
            Dict['subject'].append(pat)
            for _dir, key, el in zip(['scans'] + ['labels'] * 6, labels, elements):
                Dict[key].append('{}/img_pat_{}_{}_slice_{}.pkl'.format(_dir, pat, el, slice))
    pd.DataFrame(Dict).to_csv(DATASETPATH / '2D'/ 'dataset.csv', index = False)

def test():
    print('root_dir', root_dir)
    print(DATASETPATH, 'is a dir:', os.path.isdir(DATASETPATH))
    print(DATASETPATH / 'dataset.csv', 'is a file:', os.path.isfile(DATASETPATH / 'dataset.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--test", action='store_true', default=False)
    args = parser.parse_args()
    if args.test:
        test()
    else:
        main()

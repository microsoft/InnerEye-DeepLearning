#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
import sys
current_file = Path(__file__)
root_dir = current_file.parent.parent.parent
sys.path.append(str(root_dir / 'InnerEye-Generative'))
from locations import DATASETPATH
DATASETPATH = Path(DATASETPATH)

def main():
    df = pd.read_csv(DATASETPATH / 'dataset.csv')
    df_2D = pd.read_csv(DATASETPATH / '2D' / 'dataset.csv')
    pats = df.subject.unique()
    n_pats = len(pats)
    shuffled_pats = pats.copy()
    np.random.shuffle(shuffled_pats)
    # print(n_pats, len(shuffled_pats), len(shuffled_pats[: int(n_pats * .8)]), len(shuffled_pats[int(n_pats * .8):]))
    df_train = df.loc[df.subject.isin(shuffled_pats[: int(n_pats * .8)])]
    df_val = df.loc[df.subject.isin(shuffled_pats[int(n_pats * .8): int(n_pats * .9)])]
    df_test = df.loc[df.subject.isin(shuffled_pats[int(n_pats * .9):])] 
    # print(len(df_train), len(df_val), len(df_test))
    df_train.to_csv(DATASETPATH / 'dataset_train_80.csv')
    df_test.to_csv(DATASETPATH / 'dataset_val_10.csv')
    df_test.to_csv(DATASETPATH / 'dataset_test_10.csv')

    # on the saved 2D images
    df_2D.loc[df_2D.subject.isin(df_train.subject), 'set'] = 'train'
    df_2D.loc[df_2D.subject.isin(df_val.subject), 'set'] = 'val'
    df_2D.loc[df_2D.subject.isin(df_test.subject), 'set'] = 'test'
    # print(len(df_files), \
    #     len(df_files.loc[df_files.subject.isin(df_train.subject)]), \
    #     len(df_files.loc[df_files.subject.isin(df_val.subject)]), \
    #     len(df_files.loc[df_files.subject.isin(df_test.subject)]))
    df_2D.to_csv(DATASETPATH / '2D' / 'dataset.csv', index=False)


if __name__ == '__main__':
    main()

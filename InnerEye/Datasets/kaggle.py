#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from rich.progress import Progress

from InnerEye.ML.common import DATASET_CSV_FILE_NAME


class InnerEyeKaggleApi(KaggleApi):
    """
    Subclass as the official Kaggle API as it uses tqdm which we do not support.
    """

    def download_file(self, response, outfile, quiet=True, chunk_size=1048576) -> None:
        """
        download a file to an output file based on a chunk size

        Parameters
        ==========
        response: the response to download
        outfile: the output file to download to
        quiet: suppress verbose output (default is True)
        chunk_size: the size of the chunk to stream
        """

        outpath = os.path.dirname(outfile)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        size = int(response.headers['Content-Length'])
        size_read = 0
        if not quiet:
            print('Downloading ' + os.path.basename(outfile) + ' to ' +
                  outpath)
        with Progress() as pbar:
            task = pbar.add_task("[red]Downloading...", total=size)
            with open(outfile, 'wb') as out:
                while True:
                    data = response.read(chunk_size)
                    if not data:
                        break
                    out.write(data)
                    size_read = min(size, size_read + chunk_size)
                    pbar.update(task, advance=len(data), visible=True)
            if not quiet:
                print('\n', end='')


class KaggleDataset:
    """
    Dataset downloader to donwload datasets from Kaggle.
    """

    def __init__(self, dataset_id: str, outputs_dir: Path):
        self.dataset_id = dataset_id
        self.dst = outputs_dir / str(self.dataset_id).split("/")[-1]
        self.api = InnerEyeKaggleApi()
        self.api.authenticate()

    def download_and_prepare(self, force: bool = False) -> Path:
        """
        Download and pre-process the dataset
        force: force the download if the file already exists (default False)
        """
        if not self.dst.exists() or force:
            self.api.dataset_download_files(self.dataset_id, str(self.dst), unzip=True, quiet=False)
        self.prepare()
        return self.dst

    def prepare(self) -> None:
        raise ValueError("prepare must be defined by child classes")


class MedMNISTDatasetDownloader(KaggleDataset):
    """
    Med MNIST dataset downloader
    """

    def __init__(self, outputs_dir: Path):
        super().__init__("andrewmvd/medical-mnist", outputs_dir)

    def prepare(self) -> None:
        dataset_df = pd.DataFrame()
        classes = ["AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"]
        files = list(self.dst.rglob("*.jpeg"))
        for f in files:
            dataset_df = dataset_df.append({
                "SubjectID": f.stem,
                "channel": "image",
                "path": f.relative_to(self.dst),
                "value": classes.index(f.parent.name)
            }, ignore_index=True)
        dataset_df.to_csv(self.dst / DATASET_CSV_FILE_NAME, index=False)


class KaggleDataset(Enum):
    MedMNIST = "andrewmvd/medical-mnist"

    def create_downloader(self, outputs_dir: Path) -> KaggleDataset:
        if self == KaggleDataset.MedMNIST:
            return MedMNISTDatasetDownloader(outputs_dir)
        else:
            raise NotImplementedError(f"create_downloader is not implemented for {self.value}")

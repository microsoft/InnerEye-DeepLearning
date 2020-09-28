#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
from enum import Enum, unique
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from rich.progress import Progress


class InnerEyeKaggleApi(KaggleApi):
    """
    Hotfix as the official Kaggle API as it uses tqdm which we do not support.
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


@unique
class KaggleDataset(Enum):
    MedMNIST = "andrewmvd/medical-mnist"


class KaggleDatasetDownloader:
    def __init__(self, dataset: KaggleDataset, outputs_dir: Path):
        if dataset not in [KaggleDataset.MedMNIST]:
            raise ValueError(f"pre_process_dataset is not defined for dataset : {dataset.value}")

        self.dataset = dataset
        self.outputs_dir = outputs_dir
        self.api = InnerEyeKaggleApi()
        self.api.authenticate()

    def download_and_pre_process(self) -> Path:
        dst = self.outputs_dir / str(self.dataset.value).split("/")[-1]
        self.api.dataset_download_files(self.dataset.value, str(dst), unzip=True, quiet=False)
        self._pre_process_dataset()
        return dst

    def _pre_process_dataset(self) -> None:
        if self.dataset == KaggleDataset.MedMNIST:
            self._pre_process_med_mnist()

    def _pre_process_med_mnist(self) -> None:
        pass


if __name__ == '__main__':
    KaggleDatasetDownloader(dataset=KaggleDataset.MedMNIST, outputs_dir=Path(r"C:/")).download_and_pre_process()

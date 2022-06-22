#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
from _pytest.monkeypatch import MonkeyPatch
from fastmri.data import SliceDataset
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import VarNetDataTransform
from fastmri.pl_modules import FastMriDataModule
from pytorch_lightning import LightningDataModule, LightningModule

from InnerEye.ML.configs.other.fastmri_varnet import VarNetWithImageLogging
from InnerEye.ML.lightning_container import LightningContainer


def create_temp_data(path: Path) -> Dict[str, Any]:
    rg = np.random.default_rng(seed=1234)
    max_num_slices = 15
    max_num_coils = 15
    data_splits = {
        "knee_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
            "singlecoil_train",
            "singlecoil_val",
            "singlecoil_test",
            "singlecoil_challenge",
        ],
        "brain_data": [
            "multicoil_train",
            "multicoil_val",
            "multicoil_test",
            "multicoil_challenge",
        ],
    }

    enc_sizes = {
        "train": [(1, 128, 64), (1, 128, 49), (1, 150, 67)],
        "val": [(1, 128, 64), (1, 170, 57)],
        "test": [(1, 128, 64), (1, 96, 96)],
        "challenge": [(1, 128, 64), (1, 96, 48)],
    }
    recon_sizes = {
        "train": [(1, 64, 64), (1, 49, 49), (1, 67, 67)],
        "val": [(1, 64, 64), (1, 57, 47)],
        "test": [(1, 64, 64), (1, 96, 96)],
        "challenge": [(1, 64, 64), (1, 48, 48)],
    }

    metadata = {}
    for dataset in data_splits:
        for split in data_splits[dataset]:
            fcount = 0
            (path / dataset / split).mkdir(parents=True)
            encs = enc_sizes[split.split("_")[-1]]
            recs = recon_sizes[split.split("_")[-1]]
            for i in range(len(encs)):
                fname = path / dataset / split / f"file{fcount}.h5"
                num_slices = rg.integers(2, max_num_slices)
                if "multicoil" in split:
                    num_coils = rg.integers(2, max_num_coils)
                    enc_size = (num_slices, num_coils, encs[i][-2], encs[i][-1])  # type: ignore
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])
                else:
                    enc_size = (num_slices, encs[i][-2], encs[i][-1])  # type: ignore
                    recon_size = (num_slices, recs[i][-2], recs[i][-1])

                data = rg.normal(size=enc_size) + 1j * rg.normal(size=enc_size)

                if split.split("_")[-1] in ("train", "val"):
                    recon = np.absolute(rg.normal(size=recon_size)).astype(
                        np.dtype("<f4")
                    )
                else:
                    mask = rg.integers(0, 2, size=recon_size[-1]).astype(bool)

                with h5py.File(fname, "w") as hf:
                    hf.create_dataset("kspace", data=data.astype(np.complex64))
                    if split.split("_")[-1] in ("train", "val"):
                        hf.attrs["max"] = recon.max()
                        if "singlecoil" in split:
                            hf.create_dataset("reconstruction_esc", data=recon)
                        else:
                            hf.create_dataset("reconstruction_rss", data=recon)
                    else:
                        hf.create_dataset("mask", data=mask)

                enc_size = encs[i]  # type: ignore

                enc_limits_center = enc_size[1] // 2 + 1
                enc_limits_max = enc_size[1] - 2

                padding_left = enc_size[1] // 2 - enc_limits_center
                padding_right = padding_left + enc_limits_max

                metadata[str(fname)] = (
                    {
                        "padding_left": padding_left,
                        "padding_right": padding_right,
                        "encoding_size": enc_size,
                        "recon_size": recon_size,
                    },
                    num_slices,
                )

                fcount += 1

    return metadata


class FastMriRandomData(FastMriDataModule):
    def __init__(self) -> None:
        data_path = Path.cwd() / "data"
        if data_path.is_dir():
            shutil.rmtree(str(data_path))
        data_path.mkdir(exist_ok=False, parents=True)
        metadata = create_temp_data(data_path)

        def retrieve_metadata_mock(a: Any, fname: Any) -> Any:
            return metadata[str(fname)]

        # That's a bit flaky, we should be un-doing that after, but there's no obvious place of doing so.
        MonkeyPatch().setattr(SliceDataset, "_retrieve_metadata", retrieve_metadata_mock)

        mask = create_mask_for_mask_type(mask_type_str="equispaced",
                                         center_fractions=[0.08],
                                         accelerations=[4])
        # use random masks for train transform, fixed masks for val transform
        train_transform = VarNetDataTransform(mask_func=mask, use_seed=False)
        val_transform = VarNetDataTransform(mask_func=mask)
        test_transform = VarNetDataTransform()

        FastMriDataModule.__init__(self,
                                   data_path=data_path / "knee_data",
                                   challenge="multicoil",
                                   train_transform=train_transform,
                                   val_transform=val_transform,
                                   test_transform=test_transform)

    def prepare_data(self, *args: Any, **kwargs: Any) -> None:
        print("FastMriRandomData.prepare_data")

    def setup(self, stage: Optional[str] = None) -> None:
        print("FastMriRandomData.setup")


class FastMriOnRandomData(LightningContainer):
    def __init__(self) -> None:
        super().__init__()
        self.num_epochs = 1
        # Restrict to a single GPU, because we have code in dataset creation that could cause race conditions
        self.max_num_gpus = 1

    def create_model(self) -> LightningModule:
        return VarNetWithImageLogging()

    def get_data_module(self) -> LightningDataModule:
        # Local_dataset is set via the commandline to a random folder for unit testss
        return FastMriRandomData()

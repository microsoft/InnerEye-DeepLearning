#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from yacs.config import CfgNode
from InnerEyeDataQuality.configs.config_node import ConfigNode

from default_paths import EXPERIMENT_DIR

def create_folder(input_path: Path) -> None:
    Path(input_path).mkdir(parents=True, exist_ok=True)


def convert_labels_to_one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Converts class label ids to one-hot representation
    """
    one_hot_labels = np.zeros(shape=(labels.shape[0], n_classes), dtype=np.int64)
    one_hot_labels[np.arange(labels.shape[0]), labels.reshape(-1)] = 1
    # make sure there is exactly one label for each sample
    assert all(np.sum(one_hot_labels, axis=1) == 1)
    return one_hot_labels


def find_set_difference_torch(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Returns set difference between t1 - t2.
    """
    set_diff_np = np.setdiff1d(t1.cpu().numpy(), t2.cpu().numpy())
    return torch.from_numpy(set_diff_np).to(t1.device)


def find_common_elements_torch(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Finds the common list of elements between input tensors and returns them as torch tensor.
    """
    if (t1.nelement() == 0) | (t2.nelement() == 0):
        return torch.empty(0, dtype=t1.dtype, device=t1.device)

    indices = torch.zeros_like(t1, dtype=torch.bool, device=t1.device)
    for elem in t2:
        indices = indices | (t1 == elem)
    return t1[indices]


def find_union_set_torch(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """
    Returns the union set of elements from tensors t1 and t2.
    """
    if (t1.nelement() == 0) | (t2.nelement() == 0):
        return torch.empty(0, dtype=t1.dtype, device=t1.device)
    union_set = torch.unique(torch.cat([t1, t2], dim=0))
    return union_set


def get_logger(log_path: Path) -> None:
    logging.basicConfig(filename=str(log_path),
                        filemode='w',
                        format='%(asctime)s %(name)-4s %(levelname)-6s %(message)s',
                        datefmt='%m-%d %H:%M',
                        level=logging.DEBUG)

    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-4s: %(levelname)-6s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger().addHandler(console)


def get_data_selection_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Execute benchmark 2')
    parser.add_argument('--config', dest='config', type=str, required=True, nargs='+',
                        help='Path to config file characterising trained CNN model/s')
    parser.add_argument('--seeds', dest='seeds', required=True, type=int, nargs='+',
                        help="List of random seeds for the data selection simulation process")
    parser.add_argument('--plot-embeddings', dest='plot_embeddings', type=bool, default=False,
                        help='Flag to plot model embeddings with TSNE')
    parser.add_argument('--on-val-set', dest='on_val_set', default=False, action="store_true",
                        help='Flag to specify to run selection on training or validation set')
    parser.add_argument('--debug', dest='debug', default=False, action="store_true",
                        help='Flag to debug data selector algorithms - parallel for loop is disabled in debug mode')
    return parser

def get_data_curation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Execute data curation')
    parser.add_argument('--config', dest='config', type=str, required=True, nargs='+',
                        help='Path to config file characterising trained CNN model/s')
    return parser


def map_to_device(inputs: Union[torch.Tensor, List[torch.Tensor]],
                  device: torch.device) -> Union[torch.Tensor, List[torch.Tensor]]:
    if isinstance(inputs, list):
        return [_input.to(device) for _input in inputs]
    else:
        return inputs.to(device)


def setup_cudnn(config: CfgNode) -> None:
    torch.backends.cudnn.benchmark = config.cudnn.benchmark
    torch.backends.cudnn.deterministic = config.cudnn.deterministic


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def save_obj(obj: object, save_path: Path) -> None:
    with open(str(save_path), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(file_path: Path) -> object:
    with open(str(file_path), 'rb') as f:
        return pickle.load(f)


def get_train_output_dir(config: ConfigNode) -> str:
    """
    Returns default path to training checkpoint/tf-events output directory
    """
    config_output_dir = config.train.output_dir
    train_output_dir = EXPERIMENT_DIR / config_output_dir / f'seed_{config.train.seed:d}'
    return str(train_output_dir)

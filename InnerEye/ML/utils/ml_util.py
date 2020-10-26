#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

from InnerEye.Common import common_util
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode


@dataclass(frozen=True)
class RandomStateSnapshot:
    """
    Snapshot of all of the random generators states: python, numpy, torch.random, and torch.cuda for all gpus.
    """
    random_state: Any
    numpy_random_state: Any
    torch_random_state: Any
    torch_cuda_random_state: Any

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self, ignore=["torch_cuda_random_state"])

    @staticmethod
    def snapshot_random_state() -> RandomStateSnapshot:
        """
        Get a snapshot of all random generators state.
        """
        cuda_state = torch.cuda.get_rng_state_all() if is_gpu_available() else None  # type: ignore
        return RandomStateSnapshot(
            random_state=copy.deepcopy(random.getstate()),
            numpy_random_state=copy.deepcopy(np.random.get_state()),
            torch_random_state=copy.deepcopy(torch.random.get_rng_state()),
            torch_cuda_random_state=copy.deepcopy(cuda_state)
        )

    def restore_random_state(self) -> None:
        """
        Restore the state for the random number generators of python, numpy, torch.random, and torch.cuda for all gpus.
        """
        logging.debug("Restoring all random states")
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        if is_gpu_available():
            torch.cuda.set_rng_state_all(self.torch_cuda_random_state)  # type: ignore


def string_to_integer_list(csv_list: str, length: int) -> List[int]:
    """
    Helper function to convert an array or list of integers saved as a list in a csv back to its original format.

    :param length: excepted length of converted list
    :param csv_list: list as string
    :return: converted list of integers
    """
    if not isinstance(csv_list, str):
        raise ValueError("conversion unsuccessful, please input a string")
    if "," in csv_list:
        delimiter = ','
    else:
        delimiter = " "
    integer_list = [int(s.strip()) for s in csv_list.strip("[]").split(delimiter) if s.strip().isdigit()]
    if len(integer_list) < length:
        raise ValueError("conversion unsuccessful")
    return integer_list


def validate_dataset_paths(dataset_path: Path = Path.cwd()) -> None:
    """
    Validates that a dataset.csv file exists in the given path.

    :param dataset_path: The base path
    :raise ValueError if the dataset does not exist.
    """
    if not dataset_path.is_dir():
        raise ValueError("The dataset_path argument should be the path to the base directory of the data "
                         f"(dataset_path: {dataset_path})")
    dataset_csv = dataset_path / DATASET_CSV_FILE_NAME
    if not dataset_csv.is_file():
        raise ValueError(f"The dataset file {DATASET_CSV_FILE_NAME} file is not present in {dataset_path}")


def check_size_matches(arg1: Union[np.ndarray, torch.Tensor],
                       arg2: Union[np.ndarray, torch.Tensor],
                       dim1: int = 0,
                       dim2: int = 0,
                       matching_dimensions: Optional[List[int]] = None,
                       arg1_name: str = "arg1",
                       arg2_name: str = "arg2") -> None:
    """
    Checks if the two given numpy arrays have matching shape. Raises a ValueError if the shapes do not match.
    The shape check can be restricted to a given subset of dimensions.

    :param arg1: The first array to check.
    :param arg2: The second array to check.
    :param dim1: The expected number of dimensions of arg1. If zero, no check for number of dimensions will be
    conducted.
    :param dim2: The expected number of dimensions of arg2. If zero, no check for number of dimensions will be
    conducted.
    :param matching_dimensions: The dimensions along which the two arguments have to match. For example, if
    arg1.ndim==4 and arg2.ndim==5, matching_dimensions==[3] checks if arg1.shape[3] == arg2.shape[3].
    :param arg1_name: If provided, all error messages will use that string to instead of "arg1"
    :param arg2_name: If provided, all error messages will use that string to instead of "arg2"
    :raise ValueError if shapes don't match
    """
    if arg1 is None or arg2 is None:
        raise Exception("arg1 and arg2 cannot be None.")

    dim1 = len(arg1.shape) if dim1 <= 0 else dim1
    dim2 = len(arg2.shape) if dim2 <= 0 else dim2

    def check_dim(expected: int, actual_shape: Any, name: str) -> None:
        """
        Check if actual_shape is equal to the expected shape
        :param expected: expected shape
        :param actual_shape:
        :param name: variable name
        :raise ValueError if not the same shape
        """
        if len(actual_shape) != expected:
            raise ValueError("'{}' was expected to have ndim == {}, but is {}. Shape is {}"
                             .format(name, expected, len(actual_shape), actual_shape))

    check_dim(dim1, arg1.shape, arg1_name)
    check_dim(dim2, arg2.shape, arg2_name)
    if dim1 != dim2 and matching_dimensions is None:
        raise ValueError("When the arguments have different ndim, the 'match_dimensions' argument must be given.")
    if matching_dimensions is None:
        matching_dimensions = [i for i in range(dim1)]
    shape1 = [arg1.shape[i] for i in matching_dimensions]
    shape2 = [arg2.shape[i] for i in matching_dimensions]
    if shape1 != shape2:
        raise ValueError("Expected that '{}' and '{}' match along dimensions {}, but got: '{}'.shape == {}, "
                         "'{}'.shape == {} ".format(arg1_name, arg1_name, matching_dimensions, arg1_name,
                                                    arg1.shape, arg2_name, arg2.shape))


def set_random_seed(random_seed: int, caller_name: Optional[str] = None) -> None:
    """
    Set the seed for the random number generators of python, numpy, torch.random, and torch.cuda for all gpus.
    :param random_seed: random seed value to set.
    :param caller_name: name of the caller for logging purposes.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if is_gpu_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(random_seed)  # type: ignore
    prefix = ""
    if caller_name is not None:
        prefix = caller_name + ": "
    logging.debug(f"{prefix}Random seed set to: {random_seed}")


# noinspection PyUnresolvedReferences,PyTypeHints
def make_pytorch_reproducible() -> None:
    """
    Sets pytorch to a state such that 2 independent training runs do really give the same results.
    """
    # These two settings come from https://pytorch.org/docs/stable/notes/randomness.html
    # Caveat: small increase in training time. For classification models, training time increased from
    # 22:25min / 22:35min to 22:45min / 22:50min
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore


def is_test_from_execution_mode(execution_mode: ModelExecutionMode) -> bool:
    """
    Returns a boolean by checking the execution type. The output is used to determine the properties
    of the forward pass, e.g. model gradient updates or metric computation.
    :return True if execution mode is VAL or TEST, False if TRAIN
    :raise ValueError if the execution mode is invalid
    """
    if execution_mode == ModelExecutionMode.TRAIN:
        return False
    elif (execution_mode == ModelExecutionMode.VAL) or (execution_mode == ModelExecutionMode.TEST):
        return True
    else:
        raise ValueError("Unknown execution mode: '{}'".format(execution_mode))


def is_gpu_available() -> bool:
    """
    Returns True if a GPU with at least 1 device is available.
    """
    return torch.cuda.is_available() and torch.cuda.device_count() > 0


def is_tensor_nan_or_inf(tensor: torch.Tensor) -> bool:
    """
    Returns True if any of the tensor elements is Not a Number or Infinity.

    :param tensor: The tensor to check.
    :return: True if any of the tensor elements is Not a Number or Infinity, False if all entries are valid numbers.
    """
    result = torch.isnan(tensor).any().item() or torch.isinf(tensor).any().item()
    if isinstance(result, bool):
        return result
    raise ValueError("torch not returning bool as we expected")


def is_tensor_nan(tensor: torch.Tensor) -> bool:
    """
    Returns True if any of the tensor elements is Not a Number.

    :param tensor: The tensor to check.
    :return: True if any of the tensor elements is Not a Number, False if all entries are valid numbers.
    If the tensor is empty, the function returns False.
    """
    return bool(torch.isnan(tensor).any().item())

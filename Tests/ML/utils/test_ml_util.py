#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch

from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.ml_util import RandomStateSnapshot, check_size_matches, is_tensor_nan


def test_check_size() -> None:
    """
    Test `check_size_matches` function.
    """
    a1 = np.zeros((2, 3, 4))
    a2 = np.zeros((5, 2, 3, 4))
    check_size_matches(a1, a1)
    check_size_matches(a1, a2, matching_dimensions=[-3, -2, -1])
    check_size_matches(a1, a2, dim1=3, dim2=4, matching_dimensions=[-3, -2, -1])
    check_size_matches(a1, a1, matching_dimensions=[0, 1])

    def throws(func: Callable[..., None]) -> None:
        with pytest.raises(ValueError) as e:
            func()
        print("Exception message: {}".format(e.value))

    # Can't compare arrays of different dimension
    throws(lambda: check_size_matches(a1, a2))  # type: ignore
    # a2 has wrong dimension
    throws(lambda: check_size_matches(a1, a2, dim1=3, dim2=3))  # type: ignore
    # a1 has wrong dimension
    throws(lambda: check_size_matches(a1, a2, dim1=4, dim2=4))  # type: ignore
    # a1 has wrong dimension [0]
    throws(lambda: check_size_matches(a1, a2, dim1=4, dim2=4))  # type: ignore


def test_random_state_snapshot() -> None:
    """
    Test get and reset all random states via RandomStateSnapshot classes.
    """
    def _get_random_ints_from_libs() -> Tuple[List[int], np.ndarray, torch.Tensor]:
        _python_random = [random.randint(0, 100) for _ in range(0, 20)]
        _numpy_random = np.random.randint(0, 100, 20)
        _torch_random = torch.randint(0, 100, (20, 1))
        return _python_random, _numpy_random, _torch_random

    # set the random state
    ml_util.set_random_seed(0)
    # take snapshot of the random state at it's original state
    random_state = RandomStateSnapshot.snapshot_random_state()
    # create random numbers using python, numpy, and torch
    original_python_random, original_numpy_random, original_torch_random = _get_random_ints_from_libs()
    # re-set the random state
    ml_util.set_random_seed(0)

    # validate that the current random state is accurately captured
    assert random.getstate() == random_state.random_state
    for i, x in enumerate(np.random.get_state()):
        assert np.array_equal(x, random_state.numpy_random_state[i])
    assert torch.equal(torch.random.get_rng_state(), random_state.torch_random_state)
    assert random_state.torch_cuda_random_state is None

    # change the random state
    ml_util.set_random_seed(10)
    # create random numbers using python, numpy, and torch
    new_python_random, new_numpy_random, new_torch_random = _get_random_ints_from_libs()
    # check that a new state was used to create these random numbers
    assert not new_python_random == original_python_random
    assert not np.array_equal(new_numpy_random, original_numpy_random)
    assert not torch.equal(new_torch_random, original_torch_random)

    # restore the original random stage
    random_state.restore_random_state()
    # get restored random variables
    restored_python_random, restored_numpy_random, restored_torch_random = _get_random_ints_from_libs()
    # check restored variables match the original
    assert restored_python_random == original_python_random
    assert np.array_equal(restored_numpy_random, original_numpy_random)
    assert torch.equal(restored_torch_random, original_torch_random)


def test_is_tensor_nan() -> None:
    assert not is_tensor_nan(torch.tensor([0, 1, 3]))
    assert is_tensor_nan(torch.tensor([0, np.nan, 3]))
    assert is_tensor_nan(torch.tensor([0, np.nan, np.inf]))
    assert not is_tensor_nan(torch.tensor([0, np.inf, 3]))
    assert not is_tensor_nan(torch.tensor([]))
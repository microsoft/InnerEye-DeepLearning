#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pytest

from InnerEye.Common.fixed_paths_for_tests import full_ml_test_data_path
from InnerEye.ML.utils.hdf5_util import HDF5Object
from InnerEye.ML.utils.io_util import is_hdf5_file_path

n_classes = 11
root = full_ml_test_data_path()
test_data = [
    (root / "hdf5_data" / "patient_hdf5s" / "4be9beed-5861-fdd2-72c2-8dd89aadc1ef.h5",
     '0001',
     datetime(2018, 7, 10).isoformat(),
     np.array([4, 5, 7])),
    (root / "hdf5_data" / "patient_hdf5s" / "6ceacaf8-abd2-ffec-2ade-d52afd6dd1be.h5",
     '0011',
     datetime(2011, 7, 14).isoformat(),
     np.array([4, 5, 7])),
]

test_data_single_patient_multiple_scans = [
    ([
         root / "hdf5_data" / "patient_hdf5s" / "d316cfe5-e62a-3c0e-afda-72c3cf5ea2d8.h5",
         root / "hdf5_data" / "patient_hdf5s" / "b3200426-1a58-bfea-4aba-cbacbe66ea5e.h5"],
     '1101',
     [
         datetime(2001, 12, 2).isoformat(),
         datetime(2003, 5, 5).isoformat()],
     [
         np.array([4, 5, 7]),
         np.array([4, 5, 7])])
]


@pytest.mark.parametrize("load_segmentation", [True, False])
@pytest.mark.parametrize("hdf5_path, patient_id, acquisition_date, array_shape", test_data)
def test_load_hdf5_from_file(hdf5_path: Union[str, Path],
                             patient_id: str,
                             acquisition_date: datetime,
                             array_shape: np.ndarray,
                             load_segmentation: bool) -> None:
    """
    Assert that the HDF5 object is loaded with the expected attributes.
    """
    hdf5_path = Path(hdf5_path)
    act_hdf5 = HDF5Object.from_file(hdf5_path, load_segmentation=load_segmentation)
    assert act_hdf5.patient_id == patient_id

    assert act_hdf5.acquisition_date.isoformat() == acquisition_date  # type: ignore
    assert np.array_equal(act_hdf5.volume.shape, array_shape)
    if load_segmentation:
        assert act_hdf5.segmentation is not None
        assert np.array_equal(act_hdf5.segmentation.shape, array_shape)
    else:
        assert act_hdf5.segmentation is None


@pytest.mark.parametrize("correctly_formatted, expected", [("2012-06-07T00:00:00", datetime(2012, 6, 7)),
                                                           ("1900-01-30T10:09:08", datetime(1900, 1, 30, 10, 9, 8)),
                                                           ("1701-08-15T23:59:59", datetime(1701, 8, 15, 23, 59, 59))
                                                           ])
def test_parse_acquisition_date_correctly_formatted(correctly_formatted: str, expected: datetime) -> None:
    parsed = HDF5Object.parse_acquisition_date(correctly_formatted)
    assert parsed == expected


@pytest.mark.parametrize("badly_formatted", ["2012-30-05T00:00:00", "1999-02-28", "1701-08-15T25:59:59"])
def test_parse_acquisition_date_badly_formatted(badly_formatted: str) -> None:
    parsed = HDF5Object.parse_acquisition_date(badly_formatted)
    assert parsed is None


@pytest.mark.parametrize("input", [("foo.txt", False),
                                   ("foo.gz", False),
                                   ("foo.h5.gz", True),
                                   ("foo.h5.sz", True),
                                   ("foo.h5", True),
                                   ("foo.hdf5", True),
                                   ])
def test_is_hdf5_file(input: Tuple[str, bool]) -> None:
    file, expected = input
    assert is_hdf5_file_path(file) == expected
    assert is_hdf5_file_path(Path(file)) == expected

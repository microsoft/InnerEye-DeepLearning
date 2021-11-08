#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np

import InnerEye.Common.Statistics.report_structure_extremes as rse


def test_line_for_structure() -> None:
    data = np.zeros((10, 15, 20), dtype=int)
    # Data is a cuboid with x = 3 to 7, y = 4 to 11, z = 8 to 14
    data[3:8, 4:12, 8:15] = 1
    line = rse.line_for_structure("123", "abcdefgh", "armpit.nii.gz", data)
    # Strip out extra spaces
    stripped = " ".join(line.strip().split())
    expected = "3 7 9 4 11 14 8 14 19 123 abcdefgh armpit"
    assert stripped == expected


def test_line_for_structure_with_missing() -> None:
    data = np.zeros((10, 15, 20), dtype=int)
    # Data is two cuboids: slices 8 and 9 are empty in Y dimension
    data[3:8, 4:8, 8:15] = 1
    data[3:8, 10:12, 8:15] = 1
    line = rse.line_for_structure("123", "abcdefgh", "armpit.nii.gz", data)
    # Strip out extra spaces
    stripped = " ".join(line.strip().split())
    expected = "3 7 9 4 11 14 8 14 19 123 abcdefgh armpit yMs:8-9"
    assert stripped == expected


def test_extent_list() -> None:
    # All values from 1 to 9 inclusive
    presence1 = np.arange(1, 10)
    assert rse.extent_list(presence1, 123) == ([1, 9, 123], [])
    # All values from 1 to 5 and 8 to 10, missing out 6 and 7
    presence2 = np.concatenate([np.arange(1, 6), np.arange(8, 10)])
    assert rse.extent_list(presence2, 123) == ([1, 9, 123], ["6-7"])
    # No values
    assert rse.extent_list(np.arange(0, 0), 123) == ([-1, -1, 123], [])


def test_derive_missing_ranges() -> None:
    assert rse.derive_missing_ranges(np.array([1, 2, 3])) == []
    assert rse.derive_missing_ranges(np.array([1, 2, 4])) == ["3"]
    assert rse.derive_missing_ranges(np.array([1, 2, 5, 7, 20])) == ["3-4", "6", "8-19"]

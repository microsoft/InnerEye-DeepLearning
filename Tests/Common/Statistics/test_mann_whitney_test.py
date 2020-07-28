#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List

import math

import InnerEye.Common.Statistics.mann_whitney_test as mwt


def test_mann_whitney_on_key() -> None:
    data = [float_list(range(10)), float_list(range(5)), float_list(range(20)), float_list(range(25))]
    result = mwt.mann_whitney_on_key("Imu,neck,neck", data)
    # We expect 3 comparisons, because the 5-element list should be ignored, and there are 3 unordered
    # pairs we can make from the other 3.
    assert len(result) == 3


def float_list(seq: Any) -> List[float]:
    return [float(x) for x in seq]


def test_compose_comparison_line() -> None:
    pair1 = mwt.compose_comparison_line(0, 1, "Imu,neck,neck", [0.1, 0.3], [0.2, 0.4], [0.2, 0.3], [0.1, 0.1])
    pair2 = mwt.compose_comparison_line(1, 0, "Imu,neck,neck", [0.2, 0.4], [0.1, 0.3], [0.2, 0.3], [0.1, 0.1])
    # Check we get the same result both ways round
    assert pair1 == pair2


def test_get_z_test_p_value() -> None:
    result = mwt.get_z_test_p_value(0, 1, 5, 5, [3.0, 4.0], [1.0, 1.0])
    assert abs(result - 0.05) <= 0.01
    assert mwt.get_z_test_p_value(0, 1, 5, 5, [3.0, 3.0], [1.0, 1.0]) == 0.5
    assert mwt.get_z_test_p_value(0, 1, 5, 5, [3.0, 4.0], [0.0, 0.0]) == 0.0


def test_mean_or_zero() -> None:
    assert mwt.mean_or_zero([]) == 0.0
    assert mwt.mean_or_zero([1.0]) == 1.0
    assert mwt.mean_or_zero([1.0, 2.0]) == 1.5


def test_standard_deviation_or_zero() -> None:
    assert mwt.standard_deviation_or_zero([]) == 0.0
    assert mwt.standard_deviation_or_zero([1.0]) == 0.0
    assert mwt.standard_deviation_or_zero([1.0, 2.0]) == math.sqrt(0.5)


def test_roc_value() -> None:
    assert mwt.roc_value([], []) == 0.5
    assert mwt.roc_value([], [1]) == 0.5
    assert mwt.roc_value([1], [2]) == 1.0
    assert mwt.roc_value([2], [1]) == 0.0
    assert mwt.roc_value([1], [1]) == 0.5
    assert mwt.roc_value([2], [3, 1]) == 0.5
    assert mwt.roc_value([3, 1], [1, 4]) == 0.625


def test_get_median() -> None:
    assert mwt.get_median([]) == " " * 9
    assert mwt.get_median([12.345]) == "   12.345"
    assert mwt.get_median([12345.0]) == "1.234e+04"
    assert mwt.get_median([1, 2, 4]) == "    2.000"


def test_parse_values() -> None:
    rows = ["0,Imu,neck,neck,12.34".split(","),
            "0,Isd,neck,neck,5.67".split(","),
            "1,Imu,neck,neck,23.45".split(",")]
    result = mwt.parse_values(rows)
    expected = {"Imu,neck,neck": [12.34, 23.45], "Isd,neck,neck": [5.67]}
    assert result == expected


def test_split_statistics_data_by_institutions() -> None:
    dataset_rows = ["0,,,,inst1".split(","), "1,,,,inst1".split(","), "2,,,,inst2".split(",")]
    stats_rows = [["0,zero".split(","), "1,one".split(","), "2,two".split(",")]]
    contents, header_rows = mwt.split_statistics_data_by_institutions(dataset_rows, stats_rows, count_threshold=1)
    assert header_rows == ["1: inst1 (2 items)", "2: inst2 (1 items)", ""]
    assert contents == [[["0", "zero"], ["1", "one"]], [["2", "two"]]]
    # With count threshold 2, only inst1 should survive
    contents, header_rows = mwt.split_statistics_data_by_institutions(dataset_rows, stats_rows, count_threshold=2)
    assert header_rows == ["1: inst1 (2 items)", ""]
    assert contents == [[["0", "zero"], ["1", "one"]]]
    # With count threshold 3, neither should survive
    contents, header_rows = mwt.split_statistics_data_by_institutions(dataset_rows, stats_rows, count_threshold=3)
    assert header_rows == [""]
    assert contents == []


def test_get_arguments() -> None:
    assert mwt.get_arguments(["-d", "dataset.csv", "statistics.csv"])[1] == \
           mwt.UsagePattern.SPLIT_STATISTICS_FILES_BY_INSTITUTION
    assert mwt.get_arguments(["statistics1.csv", "statistics2.csv"])[1] == mwt.UsagePattern.COMPARE_STATISTICS_FILES
    assert mwt.get_arguments(["-a", "foo", "mstats.csv"])[1] == mwt.UsagePattern.PROCESS_METRICS_ACROSS_ALL_RUNS
    assert mwt.get_arguments(["mstats.csv"])[1] == mwt.UsagePattern.PROCESS_METRICS_ACROSS_ALL_RUNS

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

import InnerEye.Common.Statistics.wilcoxon_signed_rank_test as wt


def test_calculate_statistics() -> None:
    scores1 = {'foo': 2.0, 'bar': 3.0}
    scores2 = {'foo': 2.5, 'baz': 4.0}
    stats = wt.calculate_statistics(scores1, scores2, 1.0)
    expected = {"pairs": 1,  # number of matching pairs - only "foo" is shared
                "n1": 0,  # number of pairs in which the first score is bigger
                "n2": 1,  # number of pairs in which the second score is bigger
                "median1": 2.0,
                "median2": 2.5,
                "wilcoxon_z": 1.0,
                # We don't try and predict the wilcoxon_p value, we just "expect" what comes.
                "wilcoxon_p": stats["wilcoxon_p"]}
    assert stats == expected


def test_difference_counts() -> None:
    values1 = [1.0, 2.0, 3.0, 4.0]
    values2 = [1.1, 2.0, 2.9, 4.1]
    n1, n2 = wt.difference_counts(values1, values2)
    assert n1 == 1
    assert n2 == 2


def test_get_wilcoxon_adjustment_factor() -> None:
    assert wt.get_wilcoxon_adjustment_factor("SKIN") == wt.WILCOXON_ADJUSTMENT_FACTOR["skin"]
    assert wt.get_wilcoxon_adjustment_factor("skin") == wt.WILCOXON_ADJUSTMENT_FACTOR["skin"]
    assert wt.get_wilcoxon_adjustment_factor("foo_bar") == wt.WILCOXON_ADJUSTMENT_FACTOR["DEFAULT"]


def test_evaluate_data_pair() -> None:
    scores1 = {"foo": {"1": 1.0, "2": 2.0, "3": 3.0},
               "bar": {"1": 2.0, "3": 2.0, "4": 3.0}}
    scores2 = {"foo": {"1": 1.2, "2": 1.8, "3": 3.1},
               "baz": {"1": 2.0, "3": 3.0}}
    result = wt.evaluate_data_pair(scores1, scores2, is_raw_p_value=False)
    assert list(result.keys()) == ["foo"]


def test_compose_pairwise_results() -> None:
    dct = {"foo":
               {"pairs": 10,
                "n1": 30,
                "n2": 40,
                "median1": 3,
                "median2": 4,
                "wilcoxon_z": 1.0,
                "wilcoxon_p": 0.02}}
    result = wt.compose_pairwise_result(0.05, dct, throw_on_failure=False)
    assert len(result) == 2 and result[1].endswith(" WORSE")
    with pytest.raises(ValueError):
        # Error, because two-tail test fails:
        wt.compose_pairwise_result(0.05, dct, throw_on_failure=True)
    # No error, because two-tail test does not fail:
    result = wt.compose_pairwise_result(0.03, dct, throw_on_failure=True)
    assert len(result) == 2 and not result[1].endswith(" BETTER") and not result[1].endswith("WORSE")
    # Switch over the p value:
    dct["foo"]["wilcoxon_p"] = 0.98
    result = wt.compose_pairwise_result(0.05, dct, throw_on_failure=False)
    assert len(result) == 2 and result[1].endswith(" BETTER")


def test_run_wilcoxon_test_on_data() -> None:
    data = {"A": {"foo": {"0": 0.95, "1": 0.95, "2": 0.95, "3": 0.95, "4": 0.95}},
            "B": {"foo": {"0": 0.96, "1": 0.96, "2": 0.96, "3": 0.96, "4": 0.96}},
            "C": {"foo": {"0": 0.97, "1": 0.97, "2": 0.97, "3": 0.97, "4": 0.97}}}
    result1 = "\n".join(wt.run_wilcoxon_test_on_data(data))
    # We expect all three pairwise comparisons.
    assert result1.find("Run 1: A\nRun 2: B\n") >= 0 or result1.find("Run 1: B\nRun 2: A\n") >= 0
    assert result1.find("Run 1: A\nRun 2: C\n") >= 0 or result1.find("Run 1: C\nRun 2: A\n") >= 0
    assert result1.find("Run 1: B\nRun 2: C\n") >= 0 or result1.find("Run 1: C\nRun 2: B\n") >= 0
    # When we specify "against B", there should be no comparison between A and C.
    result2 = "\n".join(wt.run_wilcoxon_test_on_data(data, against=["B"]))
    assert result2.find("Run 1: A\nRun 2: B\n") >= 0 or result2.find("Run 1: B\nRun 2: A\n") >= 0
    assert result2.find("Run 1: A\nRun 2: C\n") < 0 and result2.find("Run 1: C\nRun 2: A\n") < 0
    assert result2.find("Run 1: B\nRun 2: C\n") >= 0 or result2.find("Run 1: C\nRun 2: B\n") >= 0

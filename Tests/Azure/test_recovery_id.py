#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import pytest

from InnerEye.Azure.azure_util import split_recovery_id


@pytest.mark.parametrize(["id", "expected1", "expected2"],
                         [("foo:bar", "foo", "bar"),
                          ("foo:bar_ab_cd", "foo", "bar_ab_cd"),
                          ("a_b_c_00_123", "a_b_c", "a_b_c_00_123"),
                          ("baz_00_123", "baz", "baz_00_123"),
                          ("foo_bar_abc_123_456", "foo_bar_abc", "foo_bar_abc_123_456"),
                          # This is the run ID of a hyperdrive parent run. It only has one numeric part at the end
                          ("foo_bar_123", "foo_bar", "foo_bar_123"),
                          # This is a hyperdrive child run
                          ("foo_bar_123_3", "foo_bar", "foo_bar_123_3"),
                          ])
def test_split_recovery_id(id: str, expected1: str, expected2: str) -> None:
    """
    Check that run recovery ids are correctly parsed into experiment and run id.
    """
    assert split_recovery_id(id) == (expected1, expected2)


@pytest.mark.parametrize(["id"], [["foo:bar:baz"], ["foo_bar"]])
def test_split_recovery_id_fails(id: str) -> None:
    """
    Test that invalid run recovery ids raise an error when parsed.
    """
    with pytest.raises(ValueError):
        split_recovery_id(id)

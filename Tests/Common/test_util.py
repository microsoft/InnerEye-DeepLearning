#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pytest

from InnerEye.Common import common_util
from InnerEye.Common.common_util import check_is_any_of, get_namespace_root, is_private_field_name, namespace_to_path, \
    path_to_namespace, print_exception
from InnerEye.Common.output_directories import OutputFolderForTests
from Tests.fixed_paths_for_tests import full_ml_test_data_path, tests_root_directory

DEFAULT_ENSEMBLE_RUN_RECOVERY_ID = "dacart_local_branch_202008:HD_99938f3c-b25f-4604-bfcc-bb7c9ed3516f"
DEFAULT_RUN_RECOVERY_ID = "melanibe_setup_branch_mock:melanibe_setup_branch_mock_1592327919_0a161296"
DEFAULT_ENSEMBLE_RUN_RECOVERY_ID_NUMERIC = "dacart_local_branch_202008:1"
DEFAULT_RUN_RECOVERY_ID_NUMERIC = "melanibe_setup_branch_mock:1"

DEFAULT_MODEL_ID_NUMERIC = "BasicModel2Epochs1Channel:1"


def test_get_items_from_string() -> None:
    """
    Check items correctly extracted from string.
    """
    assert ["i", "p"] == common_util.get_items_from_string("i, ,p")
    assert ["i", "p"] == common_util.get_items_from_string("i- -p", separator="-")
    assert ["i", " ", " p"] == common_util.get_items_from_string("i, , p", remove_blanks=False)
    assert ["i", "p"] == common_util.get_items_from_string("i, , p")
    assert [] == common_util.get_items_from_string("")


class SomeSimpleClass:
    def __init__(self) -> None:
        self.int = 1
        self.float = 3.14
        self.dict = {"foo": "Bar"}
        self.str = "str"


def test_is_any_of() -> None:
    """
    Tests for check_is_any_of: checks if a string is any of the strings in a valid set.
    """
    check_is_any_of("prefix", "foo", ["foo"])
    check_is_any_of("prefix", "foo", ["bar", "foo"])
    check_is_any_of("prefix", None, ["bar", "foo", None])
    # When the value is not found, an error message with the valid values should be printed
    with pytest.raises(ValueError) as ex:
        check_is_any_of("prefix", None, ["bar", "foo"])
    assert "bar" in ex.value.args[0]
    assert "foo" in ex.value.args[0]
    assert "prefix" in ex.value.args[0]
    # The error message should also work when one of the valid values is None
    with pytest.raises(ValueError) as ex:
        check_is_any_of("prefix", "baz", ["bar", None])
    assert "bar" in ex.value.args[0]
    assert "<None>" in ex.value.args[0]
    assert "prefix" in ex.value.args[0]
    assert "baz" in ex.value.args[0]


def test_is_field_private() -> None:
    """
    Tests for is_private_field_name
    """
    assert is_private_field_name("_hello")
    assert is_private_field_name("__hello")
    assert not is_private_field_name("world")


def test_print_exception() -> None:
    """
    A test that just throws an exception, and allows to check if the diagnostics are at the right level.
    You need to inspect the test output manually.
    """
    try:
        raise ValueError("foo")
    except Exception as ex:
        print_exception(ex, "Message")


@pytest.mark.parametrize("is_external", [True, False])
def test_namespace_to_path(is_external: bool, test_output_dirs: OutputFolderForTests) -> None:
    """
    A test to check conversion between path to namespace for InnerEye and external namespaces
    """
    if is_external:
        folder_name = "logs"
        full_folder = test_output_dirs.root_dir / folder_name
        assert namespace_to_path(folder_name, root=test_output_dirs.root_dir) == full_folder
    else:
        from Tests.ML import test_data
        assert namespace_to_path(test_data.__name__, root=tests_root_directory().parent) == full_ml_test_data_path()


@pytest.mark.parametrize("is_external", [True, False])
def test_path_to_namespace(is_external: bool, test_output_dirs: OutputFolderForTests) -> None:
    """
    A test to check conversion between namespace to path for InnerEye and external namespaces
    """
    if is_external:
        folder_name = "logs"
        full_folder = test_output_dirs.root_dir / folder_name
        assert path_to_namespace(
            path=full_folder,
            root=test_output_dirs.root_dir
        ) == folder_name
    else:
        from Tests.ML import test_data
        from Tests.fixed_paths_for_tests import full_ml_test_data_path
        assert path_to_namespace(
            path=full_ml_test_data_path(),
            root=tests_root_directory().parent
        ) == test_data.__name__


def test_get_namespace_root() -> None:
    """
    Tests that namespace root is retrieved correctly.
    """
    current_namespace = "Tests.Common.test_util"
    root = get_namespace_root(current_namespace)
    assert root is not None
    assert root.exists()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
Global PyTest configuration -- used to define global fixtures for the entire test suite

DO NOT RENAME THIS FILE: (https://docs.pytest.org/en/latest/fixture.html#sharing-a-fixture-across-tests-in-a-module
-or-class-session)
"""
import uuid
from typing import Generator, List

import pytest
from _pytest.python import Function

from InnerEye.Common.output_directories import OutputFolderForTests, remove_and_create_folder
from InnerEye.Common.fixed_paths_for_tests import TEST_OUTPUTS_PATH

NO_MARK = "no_mark"


@pytest.fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    # create a default outputs root for all tests
    remove_and_create_folder(TEST_OUTPUTS_PATH)
    # run the entire test suite
    yield


@pytest.fixture
def test_output_dirs() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test and then
    removing this directory after the test has been executed.
    """
    # create dirs before executing the test
    root_dir = TEST_OUTPUTS_PATH / str(uuid.uuid4().hex)
    remove_and_create_folder(root_dir)
    print(f"Created temporary folder for test: {root_dir}")
    # let the test function run
    yield OutputFolderForTests(root_dir=root_dir)


def pytest_collection_modifyitems(items: List[Function]) -> None:
    """
    Add a marker "no_mark" to all tests that do not have a pytest marker. From https://stackoverflow.com/a/55921954
    """
    for item in items:
        if not list(item.iter_markers()):
            item.add_marker(NO_MARK)

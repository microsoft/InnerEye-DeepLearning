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
from typing import Generator

import pytest

from InnerEye.Common.output_directories import TestOutputDirectories, make_test_output_dir
from Tests.fixed_paths_for_tests import TEST_OUTPUTS_PATH


@pytest.yield_fixture(autouse=True, scope='session')
def test_suite_setup() -> Generator:
    # create a default outputs root for all tests
    make_test_output_dir(TEST_OUTPUTS_PATH)
    # run the entire test suite
    yield


@pytest.fixture
def test_output_dirs() -> Generator:
    """
    Fixture to automatically create a random directory before executing a test and then
    removing this directory after the test has been executed.
    """
    # create dirs before executing the test
    root_dir = make_output_dirs_for_test()
    print(f"Created temporary folder for test: {root_dir}")
    # let the test function run
    yield TestOutputDirectories(root_dir=root_dir)


def make_output_dirs_for_test() -> str:
    """
    Create a random output directory for a test inside the global test outputs root.
    """
    test_output_dir = TEST_OUTPUTS_PATH / str(uuid.uuid4().hex)
    make_test_output_dir(test_output_dir)

    return str(test_output_dir)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import pytest
from typing import Optional
from InnerEye.Common import fixed_paths_for_tests
from InnerEye.ML.configs.other.HelloContainer import HelloDataModule


def test_cross_validation_splits() -> None:
    '''
    Test that the exemplar HelloDataModule correctly splits the data set for cross validation.

    We test that unreasonable values raise an exception (i.e. a split index greater than the number of splits,
    or requesting a number of splits larger than the available data).
     - We test that between splits the train and validation data differ, and the test data remains the same.
     - We test that all the data is used in each split.
     - We test that across all the splits the validation data use all of the non-test data.
    '''
    # Get full data-set for comparison
    root_folder = fixed_paths_for_tests.full_ml_test_data_path()
    data_module_no_xval = HelloDataModule(root_folder=root_folder)

    for number_of_cross_validation_splits in [0, 1, 5]:
        previous_data_module_xval: Optional[HelloDataModule] = None
        for cross_validation_split_index in range(number_of_cross_validation_splits):
            data_module_xval = HelloDataModule(
                root_folder=root_folder,
                cross_validation_split_index=cross_validation_split_index,
                number_of_cross_validation_splits=number_of_cross_validation_splits)
            _assert_total_data_identical(
                data_module_no_xval,
                data_module_xval,
                f"Total data mismatch for cross validation split ({number_of_cross_validation_splits}, {cross_validation_split_index})")
            if number_of_cross_validation_splits <= 1:
                break
            if previous_data_module_xval:
                _check_train_val_test(previous_data_module_xval, data_module_xval)
            previous_data_module_xval = data_module_xval
            if cross_validation_split_index == 0:
                accrued_val_data = data_module_xval.val.data
            else:
                accrued_val_data = torch.cat((accrued_val_data, data_module_xval.val.data), dim=0)
            if cross_validation_split_index == number_of_cross_validation_splits - 1:
                all_non_test_data = torch.cat((data_module_xval.train.data, data_module_xval.val.data), dim=0)
                msg = "Accrued validation sets from all the cross validations does not match the total non-test data"
                assert torch.equal(torch.sort(all_non_test_data, dim=0)[0], torch.sort(accrued_val_data, dim=0)[0]), msg

    with pytest.raises(IndexError) as index_error:
        data_module_xval = HelloDataModule(
            root_folder=root_folder,
            cross_validation_split_index=6,
            number_of_cross_validation_splits=5)
    assert "too large given the number_of_cross_validation_splits" in str(index_error.value)

    with pytest.raises(ValueError) as value_error:
        data_module_xval = HelloDataModule(
            root_folder=root_folder,
            cross_validation_split_index=0,
            number_of_cross_validation_splits=10_000)
    assert "Asked for 10000 cross validation splits from a dataset of length" in str(value_error.value)


def _assert_total_data_identical(dm1: HelloDataModule, dm2: HelloDataModule, msg: str) -> None:
    '''
    Check that the total of the two HelloDataModule's train, val, and test data is identical
    '''
    all_data1 = torch.cat((dm1.train.data, dm1.val.data, dm1.test.data), dim=0)
    all_data2 = torch.cat((dm2.train.data, dm2.val.data, dm2.test.data), dim=0)
    all_data1_sorted, _ = torch.sort(all_data1, dim=0)
    all_data2_sorted, _ = torch.sort(all_data2, dim=0)
    assert torch.equal(all_data1_sorted, all_data2_sorted), msg


def _check_train_val_test(dm1: HelloDataModule, dm2: HelloDataModule) -> None:
    '''
    Check that the two HelloDataModule's train and val data is different, but that their test data is identical
    '''
    msg = "Two cross validation sets have the same training data"
    assert not torch.equal(torch.sort(dm1.train.data, dim=0)[0], torch.sort(dm2.train.data, dim=0)[0]), msg
    msg = "Two cross validation sets have the same validation data"
    assert not torch.equal(torch.sort(dm1.val.data, dim=0)[0], torch.sort(dm2.val.data, dim=0)[0]), msg
    msg = "Two cross validation sets have differing test data"
    assert torch.equal(torch.sort(dm1.test.data, dim=0)[0], torch.sort(dm2.test.data, dim=0)[0]), msg

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
import sys
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SUBJECT_HEADER
from InnerEye.ML.utils.split_dataset import DatasetSplits
from Tests.fixed_paths_for_tests import full_ml_test_data_path


def test_split_by_institution() -> None:
    """
    Test if splitting by institution is as expected
    """
    random.seed(0)
    splits = [0.5, 0.4, 0.1]
    expected_split_sizes_per_institution = [[5, 3, 2], [45, 36, 9]]
    test_data = {
        CSV_SUBJECT_HEADER: list(range(0, 100)),
        CSV_INSTITUTION_HEADER: ([0] * 10) + ([1] * 90),
        "other": list(range(0, 100))
    }

    test_df = DataFrame(test_data, columns=list(test_data.keys()))
    dataset_splits = DatasetSplits.from_institutions(
        df=test_df,
        proportion_train=splits[0],
        proportion_val=splits[1],
        proportion_test=splits[2],
        shuffle=True
    )

    train_val_test = [dataset_splits.train, dataset_splits.val, dataset_splits.test]
    # Check institution ratios are as expected
    get_number_rows_for_institution = \
        lambda _x, _i: len(_x.loc[test_df.institutionId == _i].subject.unique())

    for i, inst_id in enumerate(test_df.institutionId.unique()):
        # noinspection PyTypeChecker
        for j, df in enumerate(train_val_test):
            np.isclose(get_number_rows_for_institution(df, inst_id), expected_split_sizes_per_institution[i][j])

    # Check that there are no overlaps between the datasets
    assert not set.intersection(*[set(x.subject) for x in train_val_test])

    # check that all of the data is persisted
    datasets_df = pd.concat(train_val_test)
    pd.testing.assert_frame_equal(datasets_df.sort_values([CSV_SUBJECT_HEADER], ascending=True), test_df)


@pytest.mark.parametrize("splits", [[0, 0, 0], [1, 0, 0], [-1, 0, -1], [-1, -1, -1], [1, 1, 1]])
def test_split_by_institution_invalid(splits: List[float]) -> None:
    df1 = pd.read_csv(full_ml_test_data_path(DATASET_CSV_FILE_NAME))
    with pytest.raises(ValueError):
        DatasetSplits.from_institutions(df1, splits[0], splits[1], splits[2], shuffle=False)


def test_split_by_institution_exclude() -> None:
    """
    Test if splitting data by institution correctly handles the "exclude institution" flags.
    """
    # 40 subjects across 4 institutions
    test_data = {
        CSV_SUBJECT_HEADER: list(range(40)),
        CSV_INSTITUTION_HEADER: ["a", "b", "c", "d"] * 10,
        "other": list(range(0, 40))
    }
    df = DataFrame(test_data)
    all_inst = set(df[CSV_INSTITUTION_HEADER].unique())

    def check_inst_present(splits: DatasetSplits, expected: Set[str],
                           expected_test_set: Optional[Set[str]] = None) -> None:
        assert expected == set(splits.train[CSV_INSTITUTION_HEADER].unique())
        assert expected == set(splits.val[CSV_INSTITUTION_HEADER].unique())
        assert (expected_test_set or expected) == set(splits.test[CSV_INSTITUTION_HEADER].unique())

    # Normal functionality: all 4 institutions should be present in each of train, val, test
    splits = DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3)
    check_inst_present(splits, all_inst)
    # Exclude institution "a" from all sets
    split1 = DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, exclude_institutions=["a"])
    check_inst_present(split1, {"b", "c", "d"})

    with pytest.raises(ValueError) as ex:
        DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, exclude_institutions=["not present"])
    assert "not present" in str(ex)

    # Put "a" only into the test set:
    split2 = DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, institutions_for_test_only=["a"])
    check_inst_present(split2, {"b", "c", "d"}, all_inst)

    with pytest.raises(ValueError) as ex:
        DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, institutions_for_test_only=["not present"])
    assert "not present" in str(ex)

    forced_subjects_in_test = list(df.subject.unique())[:20]
    split3 = DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, subject_ids_for_test_only=forced_subjects_in_test)
    assert set(split3.test.subject.unique()).issuperset(forced_subjects_in_test)

    with pytest.raises(ValueError) as ex:
        DatasetSplits.from_institutions(df, 0.5, 0.2, 0.3, subject_ids_for_test_only=['999'])
    assert "not present" in str(ex)


def test_split_by_subject_ids() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids)

    for x, y in zip([splits.train, splits.test, splits.val], [train_ids, test_ids, val_ids]):
        pd.testing.assert_frame_equal(x, test_df[test_df.subject.isin(y)])


@pytest.mark.parametrize("splits", [[[], ['1'], ['2']], [['1'], [], ['2']], [[], [], ['2']]])
def test_split_by_subject_ids_invalid(splits: List[List[str]]) -> None:
    df1 = pd.read_csv(full_ml_test_data_path(DATASET_CSV_FILE_NAME), dtype=str)
    with pytest.raises(ValueError):
        DatasetSplits.from_subject_ids(df1, train_ids=splits[0], val_ids=splits[1], test_ids=splits[2])


def test_get_subject_ranges_for_splits() -> None:
    def _check_at_least_one(x: Dict[ModelExecutionMode, Set[str]]) -> None:
        assert all([len(x[mode]) >= 1] for mode in x.keys())

    proportions = [0.5, 0.4, 0.1]

    splits = DatasetSplits.get_subject_ranges_for_splits(['1', '2', '3'], proportions[0], proportions[1], proportions[2])
    _check_at_least_one(splits)

    splits = DatasetSplits.get_subject_ranges_for_splits(['1'], proportions[0], proportions[1], proportions[2])
    assert splits[ModelExecutionMode.TRAIN] == {'1'}

    population = list(map(str, range(100)))
    splits = DatasetSplits.get_subject_ranges_for_splits(population, proportions[0], proportions[1], proportions[2])
    _check_at_least_one(splits)
    assert all(
        [np.isclose(len(splits[mode]) / len(population), proportions[i]) for i, mode in enumerate(splits.keys())])


def test_get_k_fold_cross_validation_splits() -> None:
    # check the dataset splits have deterministic randomness
    for i in range(2):
        test_df, test_ids, train_ids, val_ids = _get_test_df()
        splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids)
        folds = splits.get_k_fold_cross_validation_splits(n_splits=5)
        assert len(folds) == 5
        assert all([x.test.equals(splits.test) for x in folds])
        assert all(
            [len(set(list(x.train.subject.unique()) + list(x.test.subject.unique()) + list(x.val.subject.unique()))
                 .difference(set(test_df.subject.unique()))) == 0 for x in folds])


def test_restrict_subjects1() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids).restrict_subjects("2")
    assert len(splits.train.subject.unique()) == 2
    assert len(splits.val.subject.unique()) == 2
    assert len(splits.test.subject.unique()) == 2


def test_restrict_subjects2() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids).restrict_subjects("2,,3")
    assert len(splits.train.subject.unique()) == 2
    assert len(splits.val.subject.unique()) == len(val_ids)
    assert len(splits.test.subject.unique()) == 3


def test_restrict_subjects3() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids).restrict_subjects(",0,+")
    assert len(splits.train.subject.unique()) == len(train_ids)
    assert len(splits.val.subject.unique()) == 0
    assert len(splits.test.subject.unique()) == len(test_ids) + len(val_ids)


def test_restrict_subjects4() -> None:
    test_df, test_ids, train_ids, val_ids = _get_test_df()
    splits = DatasetSplits.from_subject_ids(test_df, train_ids, test_ids, val_ids).restrict_subjects("0,0,+")
    assert len(splits.train.subject.unique()) == 0
    assert len(splits.val.subject.unique()) == 0
    assert len(splits.test.subject.unique()) == len(test_ids) + len(val_ids) + len(train_ids)


def _get_test_df() -> Tuple[DataFrame, List[str], List[str], List[str]]:
    test_data = {
        CSV_SUBJECT_HEADER: list(range(0, 100)),
        CSV_INSTITUTION_HEADER: ([0] * 10) + ([1] * 90),
        "other": list(range(0, 100))
    }
    train_ids, test_ids, val_ids = list(range(0, 50)), list(range(50, 75)), list(range(75, 100))
    test_df = DataFrame(test_data, columns=list(test_data.keys()))
    return test_df, list(map(str, test_ids)), list(map(str, train_ids)), list(map(str, val_ids))


def test_parse_and_check_restriction_pattern() -> None:
    assert DatasetSplits.parse_restriction_pattern("") == (None, None, None)
    assert DatasetSplits.parse_restriction_pattern("42") == (42, 42, 42)
    assert DatasetSplits.parse_restriction_pattern("1,2,3") == (1, 2, 3)
    assert DatasetSplits.parse_restriction_pattern("1,,3") == (1, None, 3)
    assert DatasetSplits.parse_restriction_pattern(",,3") == (None, None, 3)
    assert DatasetSplits.parse_restriction_pattern("+,0,3") == (sys.maxsize, 0, 3)
    assert DatasetSplits.parse_restriction_pattern("1,2,+") == (1, 2, sys.maxsize)
    with pytest.raises(ValueError):
        # Neither 1 nor 3 fields
        DatasetSplits.parse_restriction_pattern("1,2")
    with pytest.raises(ValueError):
        # Neither 1 nor 3 fields
        DatasetSplits.parse_restriction_pattern("1,2,3,4")
    with pytest.raises(ValueError):
        # Equivalent to "+,+,+", and we only allow one "+" field.
        DatasetSplits.parse_restriction_pattern("+")
    with pytest.raises(ValueError):
        # This would mean "move the training set to validation AND to test".
        DatasetSplits.parse_restriction_pattern("0,+,+")

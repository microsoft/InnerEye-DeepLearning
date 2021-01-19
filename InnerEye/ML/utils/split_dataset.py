#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import random
import sys
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from InnerEye.Common import common_util
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SUBJECT_HEADER


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    subject_column: str = CSV_SUBJECT_HEADER
    group_column: Optional[str] = None
    allow_empty: bool = False

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)

        def pairwise_intersection(*collections: Iterable) -> Set:
            """Returns any element that appears in more than one collection."""
            from itertools import combinations
            intersection = set()
            for col1, col2 in combinations(map(set, collections), 2):
                intersection |= col1 & col2
            return intersection

        # perform dataset split validity assertions
        unique_train, unique_test, unique_val = self.unique_subjects()
        intersection = pairwise_intersection(unique_train, unique_test, unique_val)

        if len(intersection) != 0:
            raise ValueError("Train, Test, and Val splits must have no intersection, found: {}".format(intersection))

        if self.group_column is not None:
            groups_train = self.train[self.group_column].unique()
            groups_test = self.test[self.group_column].unique()
            groups_val = self.val[self.group_column].unique()
            group_intersection = pairwise_intersection(groups_train, groups_test, groups_val)
            if len(group_intersection) != 0:
                raise ValueError("Train, Test, and Val splits must have no intersecting groups, found: {}"
                                 .format(group_intersection))

        if (not self.allow_empty) and any([len(x) == 0 for x in [unique_train, unique_val]]):
            raise ValueError("train_ids({}), val_ids({}) must have at least one value"
                             .format(len(unique_train), len(unique_val)))

    def __str__(self) -> str:
        unique_train, unique_test, unique_val = self.unique_subjects()
        return f'Train: {len(unique_train)}, Test: {len(unique_test)}, and Val: {len(unique_val)}. ' \
               f'Total subjects: {len(unique_train) + len(unique_test) + len(unique_val)}'

    def unique_subjects(self) -> Tuple[Any, Any, Any]:
        return (self.train[self.subject_column].unique(),
                self.test[self.subject_column].unique(),
                self.val[self.subject_column].unique())

    def number_of_subjects(self) -> int:
        unique_train, unique_test, unique_val = self.unique_subjects()
        return len(unique_train) + len(unique_test) + len(unique_val)

    def __getitem__(self, mode: ModelExecutionMode) -> pd.DataFrame:
        if mode == ModelExecutionMode.TRAIN:
            return self.train
        elif mode == ModelExecutionMode.TEST:
            return self.test
        elif mode == ModelExecutionMode.VAL:
            return self.val
        else:
            raise ValueError(f"Model execution mode not recognized: {mode}")

    def restrict_subjects(self, restriction_pattern: str) -> DatasetSplits:
        """
        Creates a new dataset split that has at most the specified numbers of subjects in train, validation and test
        sets respectively.
        :param restriction_pattern: a string containing zero or two commas, and otherwise digits or "+". An empty
        substring will result in no restriction for the corresponding dataset. Thus "20,,3" means "restrict to 20
        training images and 3 test images, with no restriction on validation". A "+" value means "reassign all
        images from the set(s) with a numeric count (there must be at least one) to this set". Thus ",0,+" means "leave
        the training set alone, but move all validation images to the test set", and "0,2,+" means "move
        all training images and all but 2 validation images to the test set".
        :return: A new dataset split object with (at most) the numbers of subjects specified by restrict_pattern
        """

        n_train, n_val, n_test = self.parse_restriction_pattern(restriction_pattern)

        def restrict(df: pd.DataFrame, count: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
            if count is None:  # Not specified: keep everything
                return df, df[:0]
            ids = df[self.subject_column].unique()
            if count >= len(ids):  # "+", or a large number specified
                return df, df[:0]
            keep = ids[:count]
            drop = ids[count:]
            return df[df[self.subject_column].isin(keep)], df[df[self.subject_column].isin(drop)]

        train, train_drop = restrict(self.train, n_train)
        test, test_drop = restrict(self.test, n_test)
        val, val_drop = restrict(self.val, n_val)
        if n_train == sys.maxsize:
            train = train.append(val_drop).append(test_drop)
        elif n_test == sys.maxsize:
            test = test.append(train_drop).append(val_drop)
        elif n_val == sys.maxsize:
            val = val.append(train_drop).append(test_drop)

        return DatasetSplits(train=train, test=test, val=val, subject_column=self.subject_column, allow_empty=True)

    @staticmethod
    def parse_restriction_pattern(restriction_pattern: str) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        fields = restriction_pattern.split(",")

        def int_or_none(value: str) -> Optional[int]:
            if value == "":
                return None
            elif value == "+":
                return sys.maxsize
            else:
                return int(value)

        if len(fields) == 1:
            # A single non-empty field should convert to an integer and will be applied to all three sets.
            # If the string is empty, all fields will be None and no restrictions will be applied.
            n_all = int_or_none(fields[0])
            result = n_all, n_all, n_all
        elif len(fields) == 3:
            # A sequence of three fields (separated by two commas) is applied to train, val and test respectively.
            result = tuple(int_or_none(field) for field in fields)  # type: ignore
        else:
            raise ValueError(f"restrict_pattern should have either zero or two commas: {restriction_pattern}")
        if len([x for x in result if x == sys.maxsize]) > 1:
            # It makes no sense to try to move everything to two different sets.
            raise ValueError("restrict_pattern cannot be just '+' or contain more "
                             f"than one '+' field: {restriction_pattern}")
        if sys.maxsize in result and all(x is None or x == sys.maxsize for x in result):
            # It makes no sense to move images to a set when there is no set to move them from.
            raise ValueError(
                f"restrict_pattern cannot contain '+' unless it also contains a number: {restriction_pattern}")
        return result

    @staticmethod
    def get_subject_ranges_for_splits(population: Sequence[str],
                                      proportion_train: float,
                                      proportion_test: float,
                                      proportion_val: float) \
            -> Dict[ModelExecutionMode, Set[str]]:
        """
        Get mutually exclusive subject ranges for each dataset split (w.r.t to the proportion provided)
        ensuring all sets have at least one item in them when possible.

        :param population: all subjects
        :param proportion_train: proportion for the train set.
        :param proportion_test: proportion for the test set.
        :param proportion_val: proportion for the validation set.
        :return: Train, Test, and Val splits
        """
        sum_proportions = proportion_train + proportion_val + proportion_test
        if not np.isclose(sum_proportions, 1):
            raise ValueError("proportion_train({}) + proportion_val({}) + proportion_test({}) must be ~ 1, found: {}"
                             .format(proportion_train, proportion_val, proportion_test, sum_proportions))

        if not 0 <= proportion_test < 1:
            raise ValueError("proportion_test({}) must be in range [0, 1)"
                             .format(proportion_test))

        if not all([0 < x < 1 for x in [proportion_train, proportion_val]]):
            raise ValueError("proportion_train({}) and proportion_val({}) must be in range (0, 1)"
                             .format(proportion_train, proportion_val))

        subjects_train, subjects_test, subjects_val = set(population[0:1]), \
                                                      set(population[1:2]), \
                                                      set(population[2:3])
        remaining = list(population[3:])
        if proportion_test == 0:
            remaining = list(subjects_test) + remaining
            subjects_test = set()

        subjects_train |= set(remaining[: ceil(len(remaining) * proportion_train)])
        if len(subjects_test) > 0:
            subjects_test |= set(remaining[len(subjects_train):
                                           len(subjects_train) + ceil(len(remaining) * proportion_test)])
        subjects_val |= set(remaining) - (subjects_train | subjects_test)
        result = {
            ModelExecutionMode.TRAIN: subjects_train,
            ModelExecutionMode.TEST: subjects_test,
            ModelExecutionMode.VAL: subjects_val
        }
        return result

    @staticmethod
    def _from_split_keys(df: pd.DataFrame,
                         train_keys: Sequence[str],
                         test_keys: Sequence[str],
                         val_keys: Sequence[str],
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         key_column: str,
                         subject_column: str,
                         group_column: Optional[str]) -> DatasetSplits:
        """
        Takes a slice of values from each data split train/test/val for the provided keys.

        :param df: the input DataFrame
        :param train_keys: keys for training.
        :param test_keys: keys for testing.
        :param val_keys: keys for validation.
        :param key_column: name of the column the provided keys belong to
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        train_df = DatasetSplits.get_df_from_ids(df, train_keys, key_column)
        test_df = DatasetSplits.get_df_from_ids(df, test_keys, key_column)
        val_df = DatasetSplits.get_df_from_ids(df, val_keys, key_column)

        return DatasetSplits(train=train_df, test=test_df, val=val_df,
                             subject_column=subject_column, group_column=group_column)

    @staticmethod
    def from_proportions(df: pd.DataFrame,
                         proportion_train: float,
                         proportion_test: float,
                         proportion_val: float,
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         subject_column: str = CSV_SUBJECT_HEADER,
                         group_column: Optional[str] = None,
                         shuffle: bool = True,
                         random_seed: int = 0) -> DatasetSplits:
        """
        Creates a split of a dataset into train, test, and validation set, according to fixed proportions using
        the "subject" column in the dataframe, or the group column, if given.

        :param df: The dataframe containing all subjects.
        :param proportion_train: proportion for the train set.
        :param proportion_test: proportion for the test set.
        :param subject_column: Subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :param proportion_val: proportion for the validation set.
        :param shuffle: If True the subjects in the dataframe will be shuffle before performing splits.
        :param random_seed: Random seed to be used for shuffle 0 is default.
        :return:
        """
        key_column: str = subject_column if group_column is None else group_column
        split_keys = df[key_column].unique()
        if shuffle:
            # fix the random seed so we can guarantee reproducibility when working with shuffle
            random.Random(random_seed).shuffle(split_keys)
        ranges = DatasetSplits.get_subject_ranges_for_splits(
            split_keys,
            proportion_train=proportion_train,
            proportion_val=proportion_val,
            proportion_test=proportion_test
        )
        return DatasetSplits._from_split_keys(df,
                                              list(ranges[ModelExecutionMode.TRAIN]),
                                              list(ranges[ModelExecutionMode.TEST]),
                                              list(ranges[ModelExecutionMode.VAL]),
                                              key_column=key_column,
                                              subject_column=subject_column,
                                              group_column=group_column)

    @staticmethod
    def from_subject_ids(df: pd.DataFrame,
                         train_ids: Sequence[str],
                         test_ids: Sequence[str],
                         val_ids: Sequence[str],
                         *,  # make column names keyword-only arguments to avoid mistakes when providing both
                         subject_column: str = CSV_SUBJECT_HEADER,
                         group_column: Optional[str] = None) -> DatasetSplits:
        """
        Assuming a DataFrame with columns subject
        Takes a slice of values from each data split train/test/val for the provided ids.

        :param df: the input DataFrame
        :param train_ids: ids for training.
        :param test_ids: ids for testing.
        :param val_ids: ids for validation.
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        return DatasetSplits._from_split_keys(df, train_ids, test_ids, val_ids, key_column=subject_column,
                                              subject_column=subject_column, group_column=group_column)

    @staticmethod
    def from_groups(df: pd.DataFrame,
                    train_groups: Sequence[str],
                    test_groups: Sequence[str],
                    val_groups: Sequence[str],
                    *,  # make column names keyword-only arguments to avoid mistakes when providing both
                    group_column: str,
                    subject_column: str = CSV_SUBJECT_HEADER) -> DatasetSplits:
        """
        Assuming a DataFrame with columns subject
        Takes a slice of values from each data split train/test/val for the provided groups.

        :param df: the input DataFrame
        :param train_groups: groups for training.
        :param test_groups: groups for testing.
        :param val_groups: groups for validation.
        :param subject_column: subject id column name
        :param group_column: grouping column name; if given, samples from each group will always be
            in the same subset (train, val, or test) and cross-validation fold.
        :return: Data splits with respected dataset split ids.
        """
        return DatasetSplits._from_split_keys(df, train_groups, test_groups, val_groups, key_column=group_column,
                                              subject_column=subject_column, group_column=group_column)

    @staticmethod
    def from_institutions(df: pd.DataFrame,
                          proportion_train: float,
                          proportion_test: float,
                          proportion_val: float,
                          subject_column: str = CSV_SUBJECT_HEADER,
                          shuffle: bool = True,
                          random_seed: int = 0,
                          exclude_institutions: Optional[Iterable[str]] = None,
                          institutions_for_test_only: Optional[Iterable[str]] = None,
                          subject_ids_for_test_only: Optional[Iterable[str]] = None) -> DatasetSplits:
        """
        Assuming a DataFrame with columns subject and institutionId
        Takes a slice of values from each institution based on the train/test/val proportions provided,
        such that for each institution there is at least one subject in each of the train/test/val splits.

        :param df: the input DataFrame
        :param proportion_train: Proportion of images per institution to be used for training.
        :param proportion_val: Proportion of images per institution to be used for validation.
        :param proportion_test: Proportion of images per institution to be used for testing.
        :param subject_column: Name of column containing subject id.
        :param shuffle: If True the subjects in the dataframe will be shuffle before performing splits.
        :param random_seed: Random seed to be used for shuffle 0 is default.
        :param exclude_institutions: If given, all subjects where institutionId has the given value will be
        excluded from train, test, and validation set.
        :param institutions_for_test_only: If given, all subjects where institutionId has the given value will be
        placed only in the test set.
        :param subject_ids_for_test_only: If given, all images with the provided subject Ids will be placed in the
        test set.
        :return: Data splits with respected dataset split proportions per institution.
        """
        results: Dict[ModelExecutionMode, pd.DataFrame] = {}
        institutions_for_test_only = set(institutions_for_test_only) if institutions_for_test_only else set()
        subject_ids_for_test_only = set(subject_ids_for_test_only) if subject_ids_for_test_only else set()
        exclude: Set[str] = set() if exclude_institutions is None else set(exclude_institutions)
        total_subjects = len(df[subject_column].unique())
        if total_subjects < 3:
            raise ValueError("Dataset must contain at least 3 subjects, "
                             "in order to ensure non-empty Train/Test/Val sets")

        unknown_images_for_test_only_subjects = subject_ids_for_test_only.difference(set(df[subject_column].unique()))
        if unknown_images_for_test_only_subjects:
            raise ValueError(f"Subjects {unknown_images_for_test_only_subjects} "
                             f"provided in subject_ids_for_test_only were not present in the dataset.")

        all_institutions = set(df[CSV_INSTITUTION_HEADER].unique())

        def check_set(s: Set[str], message: str) -> None:
            invalid = s - all_institutions
            if len(invalid) > 0:
                raise ValueError(f"The following institutions are given as {message}, but are not present "
                                 f"in the dataset: {invalid}")

        check_set(institutions_for_test_only, "test set only")
        check_set(exclude, "exclusions")
        for name, group in df.groupby(by=CSV_INSTITUTION_HEADER):
            if name in exclude:
                continue
            subjects_by_institution = list(group[subject_column].unique())
            logging.info(f'institutionId: {name}  has {len(subjects_by_institution)} subjects: ')
            if shuffle:
                # fix the random seed so we can guarantee reproducibility when working with shuffle
                random.Random(random_seed).shuffle(subjects_by_institution)

            if name in institutions_for_test_only:
                split_ranges = {
                    ModelExecutionMode.TEST: set(subjects_by_institution),
                    ModelExecutionMode.TRAIN: set(),
                    ModelExecutionMode.VAL: set()
                }
            else:
                # exclude images in subject_ids_for_test_only from sampling
                subjects_by_institution = [x for x in subjects_by_institution if x not in subject_ids_for_test_only]

                # sample from the allowed subject and institutions
                split_ranges = DatasetSplits.get_subject_ranges_for_splits(
                    population=subjects_by_institution,
                    proportion_train=proportion_train,
                    proportion_test=proportion_test,
                    proportion_val=proportion_val
                )

                # append the subject_ids_for_test_only to the test split
                split_ranges[ModelExecutionMode.TEST] = \
                    split_ranges[ModelExecutionMode.TEST].union(subject_ids_for_test_only)

            for mode in ModelExecutionMode:
                mode_df = group[group[subject_column].isin(split_ranges[mode])]
                if mode not in results:
                    results[mode] = mode_df
                else:
                    results[mode] = pd.concat([results[mode], mode_df])

        return DatasetSplits(train=results[ModelExecutionMode.TRAIN],
                             test=results[ModelExecutionMode.TEST], val=results[ModelExecutionMode.VAL],
                             subject_column=subject_column)

    @staticmethod
    def get_df_from_ids(df: pd.DataFrame, ids: Sequence[str],
                        subject_column: str = CSV_SUBJECT_HEADER) -> pd.DataFrame:
        return df[df[subject_column].isin(ids)]

    def get_k_fold_cross_validation_splits(self, n_splits: int, random_seed: int = 0) -> List[DatasetSplits]:
        """
        Creates K folds from the Train + Val splits.

        If a group_column has been specified, the folds will be split such that
        subjects in a group will not be separated. In this case, the splits are
        fully deterministic, and random_seed is ignored.

        :param n_splits: number of folds to perform.
        :param random_seed: random seed to be used for shuffle 0 is default.
        :return: List of K dataset splits
        """
        if n_splits <= 0:
            raise ValueError("n_splits must be >= 0 found {}".format(n_splits))

        # concatenate train and val, as training set = train + val
        cv_dataset = pd.concat([self.train, self.val])

        if self.group_column is None:  # perform standard subject-based k-fold cross-validation
            # unique subjects
            subject_ids = cv_dataset[self.subject_column].unique()
            # calculate the random split indices
            k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            folds_gen = k_folds.split(subject_ids)
        else:  # perform grouped k-fold cross-validation
            # Here we take all entries, rather than unique, to keep subjects
            # matched to groups in the resulting arrays. This works, but could
            # perhaps be improved with group-by logic...?
            subject_ids = cv_dataset[self.subject_column].values
            groups = cv_dataset[self.group_column].values
            # scikit-learn uses a deterministic algorithm for grouped splits
            # that tries to balance the group sizes in all folds
            k_folds = GroupKFold(n_splits=n_splits)
            folds_gen = k_folds.split(subject_ids, groups=groups)

        ids_from_indices = lambda indices: [subject_ids[x] for x in indices]
        # create the number of requested splits of the dataset
        return [
            DatasetSplits(train=self.get_df_from_ids(cv_dataset, ids_from_indices(train_indices), self.subject_column),
                          val=self.get_df_from_ids(cv_dataset, ids_from_indices(val_indices), self.subject_column),
                          test=self.test, subject_column=self.subject_column, group_column=self.group_column)
            for train_indices, val_indices in folds_gen]

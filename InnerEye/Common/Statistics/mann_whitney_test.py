#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
This script applies several tests, including the Mann-Whitney "U" test, to various kinds of data. There
are three usage patterns, detailed in get_arguments below.

One pattern is, given two or more statistics.csv files produced by AnalyzeDataset (in
https://github.com/microsoft/InnerEye-CreateDataset), for each statistic that occurs in two or more of them,
to calculate several statistics that indicate how different the distributions are between the two files.
A variant on this is to supply only one statistics.csv file but add a dataset.csv that details which institution
each subject comes from. The subjects are then split by institution, just as if they had been in separate
per-institution statistics.csv files. The final pattern is to run on a MetricsAcrossAllRuns.csv produced by
plot_cross_validation.py; the comparisons then are between the Dice scores achieved by different models.

Each statistics.csv input file should have lines of the form

subjectId,statisticName,structure1,structure2,value

Each row of the output will have the following values:

ROC Mann-Wht-p Normal-p I N(I) Mean(I) Std(I) J N(J) Mean(J) Std(J) Statistic

where:

Statistic is the statistic (combination of statisticName, structure1 and structure2) being compared
I and J are the indices (starting from 1) of the two input statistics.csv datasets being compared
N(I), Mean(I) and Std(I) are the count, mean and standard deviation of the statistic value in dataset I
N(J), Mean(J) and Std(J) likewise for dataset J
ROC is the Receiver Operating Characteristic (or AUC, Area Under Curve) statistic for the two sets
   of values, https://en.wikipedia.org/wiki/Receiver_operating_characteristic; mostly easily understood as
   the probability that if I choose a value v_I from dataset I uniformly at random, and another from v_J,
   then v_I is less than v_J (with ties broken by making a random choice between yes and no). So values
   near to 0.5 indicate similar distributions, while values near 0.0 or 1.0 indicate very different ones.
Mann-Wht-p is the p-value from the Mann-Whitney "U" test (https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
   that the two sets of values come from the same distribution. No assumptions are made about what distribution
   that is, i.e. this is a non-parametric test.
Normal-p is the p-value from the Z test (https://en.wikipedia.org/wiki/Z-test) which assumes the two sets of
   values are normally distributed. Often they are not, so treat with caution. However, this statistic is
   presented because it gives some insight into the significance of any differences between the two
   "Mean" values.

The results are ordered by increasing Mann-Wht-p value, so the most significant results are at the top
of the output. Comparisons are only made when there are at least ten values for the statistic in each
of sets I and J. The distribution with the (apparently) larger values is assigned to J, and other to I.

The "p" values should be interpreted with a suitable correction for the fact that we are testing large
numbers of hypotheses together; see e.g. https://en.wikipedia.org/wiki/Bonferroni_correction. For example,
if you would regard a p value of 0.05 or less as significant for testing a single hypothesis, then when
testing 100 hypotheses, your p value threshold should be 0.05/100.

Also note: when there is an underlying difference between two populations, p values of all kinds
get more extreme (lower) the more samples there are from the populations. But ROC values do not, so once you
accept that the p value is telling you there is a difference, ROC is a better measure of how well separated
the distributions are.

Also note: this script is the one to use when comparing the performance of the SAME model on test data from
two DIFFERENT sources, in order to determine whether the model does better on one test set than the other.
If you want to compare the performance of two DIFFERENT models on the SAME test data, you should use the Wilcoxon
signed-rank test instead (see wilcoxon_signed_rank_test.py), which does pairwise comparisons between scores and is
therefore more powerful than Mann-Whitney. You *could* use the Mann-Whitney test on paired data, but it would
be less powerful than the Wilcoxon test because it does not take advantage of the opportunity to pair the scores.
For example, if one model scored 0.900, 0.910, ..., 0.990 on a set of ten test subjects, and another model scored
0.901, 0.911, ..., 0.991 on the same subjects, then the Mann-Whitney test would not give a very strong result
(the two sets of scores cover very similar distributions) but the Wilcoxon test would show that the second model
is significantly better (it scores higher on each of the ten test subjects, even though only by a little).
"""

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from enum import Enum
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import math
import pandas as pd
from scipy import stats

# Minimum number of subjects for a group to be eligible for comparison
from InnerEye.Common.common_util import FULL_METRICS_DATAFRAME_FILE

MINIMUM_INSTANCE_COUNT = 10


def compose_distribution_comparisons(file_contents: List[List[List[str]]]) -> List[str]:
    """
    Composes comparisons as detailed above.
    :param file_contents: two or more lists of rows, where each "rows" is returned by read_csv_file on
    (typically) a statistics.csv file
    :return a list of lines to print
    """
    value_lists: List[Dict[str, List[float]]] = [parse_values(rows) for rows in file_contents]
    return compose_distribution_comparisons_on_lists(value_lists)


def compose_distribution_comparisons_on_lists(value_lists: List[Any]) \
        -> List[str]:
    # "List[Any]" above should be "List[Union[defaultdict[str, List[float]], Dict[str, List[float]]]]"
    # but mypy rejects that.
    key_count: DefaultDict[str, int] = defaultdict(int)
    for values in value_lists:
        for key in values:
            key_count[key] += 1
    result_pairs = []
    for key in sorted(key_count):
        if key_count[key] > 1:
            result_pairs.extend(mann_whitney_on_key(key, [vals[key] for vals in value_lists]))
    if len(result_pairs) == 0:
        return []
    header = (f"{'ROC':^6s} {'Mann-Wht p':>12s} {'Normal p':>12s} | "
              f"I {'N(I)':>5s} {'Mean(I)':>11s} {'Std(I)':>11s} | "
              f"I {'N(J)':>5s} {'Mean(J)':>11s} {'Std(J)':>11s} | Statistic")
    return [header] + [line for _, line in sorted(result_pairs)]


def mann_whitney_on_key(key: str, lists: List[List[float]]) -> List[Tuple[Tuple[float, float], str]]:
    """
    Applies Mann-Whitney test to all sets of values (in lists) for the given key,
    and return a line of results, paired with some values for ordering purposes.
    Member lists with fewer than MINIMUM_INSTANCE_COUNT items are discarded.
    :param key: statistic name; "Vol" statistics have mm^3 replaced by cm^3 for convenience.
    :param lists: list of lists of values
    """
    # Replace short sublists with empty ones to prevent result lines being calculated for them.
    lists = [lst if len(lst) >= 10 else [] for lst in lists]
    if key.startswith('Vol,'):
        # convert mm^3 to cm^3
        lists = [[item * 0.001 for item in lst] for lst in lists]
    means = [mean_or_zero(lst) for lst in lists]
    standard_deviations = [standard_deviation_or_zero(lst) for lst in lists]
    result_pairs = []
    for (i1, lst1) in enumerate(lists):
        for (i2, lst2) in enumerate(lists):
            if i2 > i1 and lst1 and lst2:
                try:
                    result_pairs.append(compose_comparison_line(i1, i2, key, lst1, lst2, means, standard_deviations))
                except ValueError:
                    pass
    return result_pairs


def compose_comparison_line(i1: int, i2: int, key: str, lst1: List[float], lst2: List[float],
                            means: List[float], standard_deviations: List[float]) -> Tuple[Tuple[float, float], str]:
    _, mann_whitney_p = stats.mannwhitneyu(lst1, lst2, alternative='two-sided')
    roc = roc_value(lst1, lst2)
    if roc > 0.5:
        i1, i2 = i2, i1
        lst1, lst2 = lst2, lst1
        roc = 1 - roc
    norm_p = get_z_test_p_value(i1, i2, len(lst1), len(lst2), means, standard_deviations)
    result = (f"{roc:6.4f} {mann_whitney_p:12.4e} {norm_p:12.4e} | "
              f"{i1 + 1:d} {len(lst1):5d} {means[i1]:11.4f} {standard_deviations[i1]:11.4f} | "
              f"{i2 + 1:d} {len(lst2):5d} {means[i2]:11.4f} {standard_deviations[i2]:11.4f} | {key}")
    return (mann_whitney_p, norm_p), result


def get_z_test_p_value(i1: int, i2: int, len1: int, len2: int, means: List[float],
                       stds: List[float]) -> float:
    if means[i1] == means[i2]:
        return 0.5
    if stds[i1] == 0 and stds[i2] == 0:
        return 0.0
    se_diff = math.sqrt(stds[i1] * stds[i1] / len1 + stds[i2] * stds[i2] / len2)
    z = abs(means[i2] - means[i1]) / se_diff
    return stats.norm.cdf(-z)


def mean_or_zero(lst: List[float]) -> float:
    if len(lst) >= 1:
        return statistics.mean(lst)
    return 0.0


def standard_deviation_or_zero(lst: List[float]) -> float:
    if len(lst) >= 2:
        return statistics.stdev(lst)
    return 0.0


def roc_value(lst1: List[float], lst2: List[float]) -> float:
    """
    :param lst1: a list of numbers
    :param lst2: another list of numbers
    :return: the proportion of pairs (x, y), where x is from lst1 and y is from lst2, for which
    x < y, with x == y counting as half an instance.
    """
    if len(lst1) == 0 or len(lst2) == 0:
        return 0.5
    pairs = sorted([(x, 1) for x in lst1] + [(x, 2) for x in lst2])
    # Number of items from lst1 seen so far
    n1 = 0
    # Number of pairs (x, y) seen so far where x is from lst1, y is from lst2, and x < y; if x == y
    # we count 0.5
    numerator = 0.0
    # The last value seen in lst1; initialized to a non-number to force non-equality
    last_value = None
    # Number of items seen in lst1 that are equal to last_value
    last_value_count = 0
    for value, idx in pairs:
        if idx == 1:
            # It's an item from lst1.
            n1 += 1
            # Update last_value and its count.
            if value == last_value:
                last_value_count += 1
            else:
                last_value = value
                last_value_count = 1
        else:
            # It's an item from lst2.
            if value != last_value:
                last_value_count = 0
            # We add on 1 for each of the n1 members of lst1 previously seen, except we only
            # add on 0.5 for last_value_count of them.
            numerator += n1 - last_value_count * 0.5
    # Number of pairs (x, y) in the data where x is from lst1 and y is from lst2, regardless of order,
    # so that returned value is normalized to [0, 1].
    denominator = len(lst1) * len(lst2)
    return numerator / denominator


def get_median(lst: List[float]) -> str:
    """
    Returns the median value in the list as a 9-character string, or nine spaces
    if the list is empty.
    """
    if len(lst) == 0:
        return f"{'':9s}"
    elif max([abs(x) for x in lst]) >= 10000:
        return f"{statistics.median(lst):9.3e}"
    else:
        return f"{statistics.median(lst):9.3f}"


def parse_values(rows: List[List[str]]) -> Dict[str, List[float]]:
    """
    Returns a dictionary whose keys are statistics (middle columns of stats.csv
    joined with commas, e.g. "Xmd,bladder,prostate"), and whose values are lists of statistic values
    (final column).
    """
    values: DefaultDict[str, List[float]] = defaultdict(list)
    for row in rows:
        try:
            key = ",".join(row[1:len(row) - 1])
            val = float(row[-1])
            values[key].append(val)
        except ValueError:
            pass
    return values


def read_csv_file(input_file: str) -> List[List[str]]:
    """
    Reads and returns the contents of a csv file. Empty rows (which can
    result from end-of-line mismatches) are dropped.
    :param input_file: path to a file in csv format
    :return: list of rows from the file
    """
    with open(input_file, "r") as fp:
        return [row for row in csv.reader(fp) if len(row) > 0]


def compare_scores_across_institutions(metrics_file: str, splits_to_use: str = "",
                                       mode_to_use: str = "test") -> List[str]:
    """
    :param metrics_file:  MetricsAcrossAllRuns.csv file produced by plot_cross_validation.py
    :param splits_to_use: a comma-separated list of split names
    :param mode_to_use: test, validation etc
    :return: a list of comparison lines between pairs of splits. If splits_to_use is non empty,
    only pairs involving at least one split from that set are compared.
    """
    valid_splits = set(splits_to_use.split(",")) if splits_to_use else None
    metrics = pd.read_csv(metrics_file)
    data: DefaultDict[str, DefaultDict[str, DefaultDict[str, List[float]]]] = \
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for tup in metrics.itertuples():
        if (valid_splits is None or tup.split in valid_splits) and tup.mode.lower() == mode_to_use:
            data[tup.split][tup.institutionId][tup.Structure].append(tup.Dice)
    lines = []
    for split in sorted(data):
        data_for_split = data[split]
        remove_rare_institutions(data_for_split)
        institution_ids = sorted(data_for_split.keys())
        value_lists = [data_for_split[inst] for inst in institution_ids]
        comparisons = compose_distribution_comparisons_on_lists(value_lists)
        if comparisons:
            lines.append(f"For split {split}:")
            for (idx, name) in enumerate(institution_ids, 1):
                lines.append(f"{idx:d}: {name:s}")
            lines.append("")
            lines.extend(comparisons)
            lines.append("")
    return lines


def remove_rare_institutions(data: DefaultDict[str, DefaultDict[str, List[float]]],
                             threshold: int = MINIMUM_INSTANCE_COUNT) -> None:
    # Only keep institutions that have at least a minimum number of scores for at least one structure
    to_remove = set(data.keys())
    for institution in data:
        for structure in data[institution]:
            if len(data[institution][structure]) >= threshold:
                to_remove.remove(institution)
                break
    for institution in to_remove:
        del data[institution]


class UsagePattern(Enum):
    SPLIT_STATISTICS_FILES_BY_INSTITUTION = 1
    COMPARE_STATISTICS_FILES = 2
    PROCESS_METRICS_ACROSS_ALL_RUNS = 3


def get_arguments(arglist: List[str] = None) -> Tuple[Optional[argparse.Namespace], Optional[UsagePattern]]:
    """
    Usage patterns:
    (1) Compare statistics from different institutions:
        mann_whitney_test.py -d dataset.csv statistics.csv
    (2) Compare statistics from different statistics.csv files:
        mann_whitney_test.py statistics1.csv statistics2.csv ...
    (3) Compare results from MetricsAcrossAllRuns.csv file (which contains institution IDs):
        mann_whitney_test.py [-a some_split_name] MetricsAcrossAllRuns.csv
        The value of the "-a" switch is one or more split names; pairs of splits not including these
        will not be compared.
    :return: parsed arguments and identifier for pattern (1, 2, 3 as above), or None, None if none of the
    patterns are followed
    """
    # Use argparse because we want to have mandatory non-switch arguments, which GenericConfig doesn't support.
    parser = argparse.ArgumentParser("Run Mann-Whitney tests")
    parser.add_argument("-d", "--dataset", default=None,
                        help="dataset.csv file for splitting a statistics file by institutions")
    parser.add_argument("-a", "--against", default=None,
                        help="name of a split to include in all pairwise comparisons, e.g. ENSEMBLE")
    parser.add_argument("files", nargs="+", help=f"statistics.csv file(s) or {FULL_METRICS_DATAFRAME_FILE}")
    args = parser.parse_args(arglist)
    if args.dataset:
        if len(args.files) < 1 or args.against is not None:
            return None, None
        return args, UsagePattern.SPLIT_STATISTICS_FILES_BY_INSTITUTION
    if len(args.files) > 1:
        if args.against is not None:
            return None, None
        return args, UsagePattern.COMPARE_STATISTICS_FILES
    if len(args.files) == 1:
        return args, UsagePattern.PROCESS_METRICS_ACROSS_ALL_RUNS
    return None, None


def split_statistics_files_by_institutions(dataset_path: str, stats_paths: List[str]) \
        -> Tuple[List[List[List[str]]], List[str]]:
    """
    Reads the dataset_path and each member of stats_paths as csv files, and passes the data on.
    """
    dataset_rows = read_csv_file(dataset_path)
    stats_rows_list = [read_csv_file(path) for path in stats_paths]
    return split_statistics_data_by_institutions(dataset_rows, stats_rows_list)


def split_statistics_data_by_institutions(dataset_rows: List[List[str]], stats_rows_list: List[List[List[str]]],
                                          count_threshold: int = MINIMUM_INSTANCE_COUNT) \
        -> Tuple[List[List[List[str]]], List[str]]:
    # Map from subjects to institution IDs
    institution: Dict[str, str] = {}
    # Map from institution IDs to lists of rows
    institution_rows: Dict[str, List[List[str]]] = {}
    for row in dataset_rows:
        if len(row) >= 5:
            institution[row[0]] = row[4]
            # Ensure this institution is recorded; we'll add its rows later
            institution_rows[row[4]] = []
    # Populate the rows for each institution
    for stats_rows in stats_rows_list:
        for row in stats_rows:
            subj = row[0]
            inst = institution.get(subj)
            if inst is None:
                raise ValueError(f"Subject {subj} is not in dataset")
            institution_rows[inst].append(row)
    # Set of all the institutions we know about
    institution_id_set = set(institution_rows)
    # Set subject count for each institution
    subject_count = {}
    for ident in institution_id_set:
        subject_count[ident] = 0
    for ident in institution.values():
        subject_count[ident] += 1
    # Forget institutions with too few subjects
    for ident in institution_rows:
        if subject_count[ident] < count_threshold:
            institution_id_set.remove(ident)
    # Construct data rows for each institution, and header rows to print
    institution_ids = sorted(institution_id_set)
    header_rows = [f"{idx}: {ident} ({subject_count[ident]} items)"
                   for (idx, ident) in enumerate(institution_ids, 1)] + [""]
    return [institution_rows[inst] for inst in institution_ids], header_rows


def main() -> None:
    """
    Main function.
    """
    args, pattern = get_arguments()
    assert args is not None  # for mypy
    file_contents: List[List[List[str]]]
    if pattern == UsagePattern.SPLIT_STATISTICS_FILES_BY_INSTITUTION:
        pair: Tuple[List[List[List[str]]], List[str]] = split_statistics_files_by_institutions(args.dataset, args.files)
        file_contents, header_lines = pair
        if len(file_contents) > 1:
            for line in header_lines + compose_distribution_comparisons(file_contents):
                print(line)
        else:
            print("Only one institution present, no comparisons to do")
    elif pattern == UsagePattern.COMPARE_STATISTICS_FILES:
        for (idx, name) in enumerate(args.files, 1):
            print(f"{idx:d}: {name:s}")
        print("")
        file_contents = [read_csv_file(path) for path in args.files]
        for line in compose_distribution_comparisons(file_contents):
            print(line)
    elif pattern == UsagePattern.PROCESS_METRICS_ACROSS_ALL_RUNS:
        for line in compare_scores_across_institutions(args.files[0]):
            print(line)
    else:
        print(f"Usage 1: {sys.argv[0]} [-d dataset.csv] statistics1.csv")
        print(f"Usage 2: {sys.argv[0]} statistics1.csv statistics2.csv ...")
        print(f"Usage 3: {sys.argv[0]} [-a split_name] {FULL_METRICS_DATAFRAME_FILE}")


if __name__ == '__main__':
    main()

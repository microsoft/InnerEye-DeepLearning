#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Performs the Wilcoxon Signed-Rank test (https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
on two or more sets of Dice scores. If there are exactly two sets, it fails (throws) if for any
structure, the scores from the second set are significantly worse (in the sense of the p value
from the Wilcoxon test being less than the specified threshold) than those in the first. If
there are more than two sets, the test is carried out on each pair unless "--against" is specified,
but no exception is thrown.

This code is normally called from plot_cross_validation.py, but can also be run from the command line
on the MetricsAcrossAllRuns.csv file that it creates. For example:

wilcoxon_signed_rank_test.py MetricsAcrossAllRuns.csv
wilcoxon_signed_rank_test.py --against ENSEMBLE MetricsAcrossAllRuns.csv

The second form compares every other run against the one named ENSEMBLE, but not those runs against each other.

Example output comparison:

Job 1: 19830
Job 2: 19840
Name         N  N1>2 N2>1  Med1  Med2  WilcZ WilcP 2_vs_1
BLADDER     84   11   73  0.957 0.961  6.132 1.000 BETTER
FEMURL      84   16   68  0.967 0.973  5.905 1.000 BETTER
FEMURR      84   16   68  0.963 0.971  5.570 1.000 BETTER
PROSTATE    84   40   44  0.868 0.863 -0.263 0.417
PROSTATESV  84   42   42  0.837 0.823 -0.778 0.267
RECTUM      84   34   50  0.784 0.796  2.199 0.961 BETTER
SKIN        84   49   34  0.996 0.996 -2.838 0.078
SV          84   40   42  0.636 0.630 -0.072 0.477

N is the number of images (Dice scores) shared between the two sets.
N1>2 is the number of those images for which the first set has a higher score.
N2>1 is the number for which the second is higher.
Med1 and Med2 are the medians of the two sets.
WilcZ is the "z" statistic calculated from the Wilcoxon test (using the normal
  assumption).
WilcP is the P value for WilcZ. Unless the "--raw" switch is given, the WilcZ value is
  "deflated" before WilcP is calculated; see WILCOXON_ADJUSTMENT_FACTOR below.
2_vs_1 is "BETTER" if set 2 scores significantly better than set 1, "WORSE" if it's
  significantly worse, and blank otherwise.

Note: this script is the one to use when comparing the performance of two (or more) DIFFERENT models on the SAME
test data, in order to determine whether one model is better (with respect to that test data) than the other(s).
If you want to compare the performance of the SAME model on DIFFERENT test sets, you should use the Mann-Whitney
U test instead (see mann_whitney_test.py). This is because the Wilcoxon test only applies when you can pair
up each score from one set of results with a score from the other set.
"""

from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import param
from azureml.core import Run
from scipy import stats

import InnerEye.Common.Statistics.statistical_tests as tests
from InnerEye.Common.common_util import FULL_METRICS_DATAFRAME_FILE
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.visualizers.metrics_scatterplot import create_scatterplots

"""
The factor by which the Wilcoxon Z value should be divided to allow for incomplete independence of the data.
Experimentation (from comparing models built from exactly the same code and data, but different random seeds)
suggests this should be about 2 for skin/external, as Dice scores for that are very high, and about 1.25 for other
structures. You can stop it being applied by specifying "--raw" in the command line.
"""
WILCOXON_ADJUSTMENT_FACTOR = {'skin': 2.0, 'external': 2.0, 'DEFAULT': 1.25}


class WilcoxonTestConfig(GenericConfig):
    """
    Command line parameter class.
    """
    against: Optional[List[str]] = param.List(class_=str, default=None,
                                              doc='Run (value in "split" column of csv_file) against which to '
                                                  'compare every other set')
    exclude: str = param.String(default='',
                                doc='comma-separated list of structure names to exclude, e.g. external,bladder')
    raw: bool = param.Boolean(default=False, doc='Use raw Wilcoxon "z" values to calculate "p", i.e. no deflation')
    threshold: float = param.Number(default=0.05, doc='(Two-tailed) p value to trigger test failure')
    subset: str = param.String(default='test',
                               doc='all, train or test: which subset of scores to include')
    csv_file: str = param.String(default=FULL_METRICS_DATAFRAME_FILE,
                                 doc='csv file, typically created by plot_validation.py')
    data: Optional[pd.DataFrame] = param.DataFrame(default=None, doc='data, as pandas dataframe')
    with_scatterplots: bool = param.Boolean(default=False, doc='whether to generate scatterplots')


def calculate_statistics(dist1: Dict[str, float], dist2: Dict[str, float], factor: float) -> Dict[str, float]:
    """
    Select common pairs and run the hypothesis test.
    :param dist1: mapping from keys to scores
    :param dist2: mapping from keys (some or all in common with dist1) to scores
    :param factor: factor to divide the Wilcoxon "z" value by to determine p-value
    :return: dictionary from summary-statistic names to values
    """
    shared = sorted(set(key for key in dist1 if key in dist2))
    values1 = [dist1[key] for key in shared]
    values2 = [dist2[key] for key in shared]
    mean1 = np.mean(values1)
    mean2 = np.mean(values2)
    median1 = np.median(values1)
    median2 = np.median(values2)
    n1, n2 = difference_counts(values1, values2)
    # We don't use the Scipy Wilcoxon test method, as it doesn't
    # tell us which way round the statistic is.
    wil_z = tests.wilcoxon_z(values1, values2)
    wil_p = stats.norm.cdf(wil_z / factor)
    return {
        "pairs": len(shared),
        "n1": n1,
        "n2": n2,
        "mean1": mean1,
        "mean2": mean2,
        "median1": median1,
        "median2": median2,
        "wilcoxon_z": wil_z,
        "wilcoxon_p": wil_p
    }


def difference_counts(values1: List[float], values2: List[float]) -> Tuple[int, int]:
    """
    Returns the number of corresponding pairs from vals1 and vals2
    in which val1 > val2 and val2 > val1 respectively.
    :param values1: list of values
    :param values2: list of values, same length as values1
    :return: number of pairs in which first value is greater than second, and number of pairs
    in which second is greater than first
    """
    n1 = 0
    n2 = 0
    for (val1, val2) in zip(values1, values2):
        d = val2 - val1
        if d > 0:
            n2 += 1
        elif d < 0:
            n1 += 1
    return n1, n2


def get_wilcoxon_adjustment_factor(structure_name: str) -> float:
    """
    Returns the factor by which the Wilcoxon Z value should be divided to allow
    for incomplete independence of the data.
    """
    return WILCOXON_ADJUSTMENT_FACTOR.get(structure_name.lower()) or WILCOXON_ADJUSTMENT_FACTOR['DEFAULT']


def evaluate_data_pair(data1: Dict[str, Dict[str, float]], data2: Dict[str, Dict[str, float]], is_raw_p_value: bool) \
        -> Dict[str, Dict[str, float]]:
    """
    Find and compare dice scores for each structure
    :param data1: dictionary from structure names, to dictionary from subjects to scores
    :param data2: another such dictionary, sharing some structure names
    :param is_raw_p_value: whether to use "raw" Wilcoxon z values when calculating p values (rather than reduce)
    :return: dictionary from structure names to dictionary from summary-statistic names to values
    """
    results = {}
    # Compare the scores for each structure name
    for name in data1:
        if name in data2:
            factor = 1 if is_raw_p_value else get_wilcoxon_adjustment_factor(name)
            results[name] = calculate_statistics(data1[name], data2[name], factor)
    return results


def compose_pairwise_result(threshold: float, results: Dict[str, Dict[str, float]], throw_on_failure: bool = False) \
        -> List[str]:
    """
    Composes results in human readable form and (if throw_on_failure) throws if any tests fail
    """
    header = [f"{'Name':<15s} {'N':^3s} {'Mean1':^7s} {'Mean2':^7s} {'N1>2':^4s} {'N2>1':^4s} "
              f"{'Med1':^7s} {'Med2':^7s} {'WilcZ':>6s} {'WilcP':^5s} {'2_vs_1':<s}"]
    n_failed = 0
    lines: List[str] = []
    test_is_valid = False
    max_len = max([0] + [len(name) for name in results])
    for name in sorted(results):
        dct = results[name]
        line = f"{name:<{max_len}s} {dct['pairs']:3d} {dct['mean1']:7.3f} {dct['mean2']:7.3f} {dct['n1']:4d} " \
               f"{dct['n2']:4d} {dct['median1']:7.3f} {dct['median2']:7.3f}"
        if dct['n1'] + dct['n2'] >= 5:
            failure = dct['wilcoxon_p'] < threshold / 2
            pf = failure and 'WORSE' or (dct['wilcoxon_p'] > 1 - threshold / 2 and 'BETTER') or ''
            if failure:
                n_failed += 1
            line += f" {dct['wilcoxon_z']:6.3f} {dct['wilcoxon_p']:5.3f} {pf}"
            test_is_valid = True
        lines.append(line)
    if throw_on_failure and n_failed > 0:
        raise ValueError(f"{n_failed} tests failed!!!")
    if test_is_valid:
        return header + lines
    return []


def read_data(csv_file: str, subset: str = 'all', exclude: Optional[List[str]] = None) \
        -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    :param csv_file: file to read
    :param subset: 'all', 'test', or 'train': which subjects to keep
    :param exclude: list of structure names to exclude; None is equivalent to an empty list
    :return: multi-level dictionary "data" such that data[build][structure][seriesId] = dice_score
    """
    csv_data = pd.read_csv(csv_file)
    return convert_data(csv_data, subset, exclude)


def convert_data(csv_data: pd.DataFrame, subset: str = 'all', exclude: Optional[List[str]] = None) -> Dict[str, Dict]:
    """
    :param csv_data: dataframe with columns "mode", "split", "Structure", "seriesId" or "Patient", and "Dice"
    :param subset: if "all", use all rows, else use only rows whose "mode" value case-insensitively equals subset
    :param exclude: list of values of "Structure": matching rows will be dropped
    :return: dictionary such that data[split][Structure][seriesId or Patient] = Dice
    """
    data: defaultdict = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    if exclude is None:
        exclude = []
    # Prefer to use seriesId to identify the patient, but if not present, use Patient instead.
    use_series_id = "seriesId" in csv_data
    subset = subset.lower()
    for tup in csv_data.itertuples():
        if subset in ['all', tup.mode.lower()] and tup.Structure not in exclude:
            data[tup.split][tup.Structure][tup.seriesId if use_series_id else tup.Patient] = float(tup.Dice)
    return data


def wilcoxon_signed_rank_test(args: WilcoxonTestConfig,
                              name_shortener: Optional[Callable[[Union[Run, str]], str]] = None) \
        -> Tuple[List[str], Dict[str, plt.figure]]:
    """
    Reads data from a csv file, and performs all pairwise comparisons, except if --against was specified,
    compare every other run against the "--against" run.
    :param args: parsed command line parameters
    :param name_shortener: optional function to shorten names to make graphs and tables more legible
    """
    if args.data is not None:
        data = convert_data(args.data)
    else:
        data = read_data(args.csv_file, args.subset, args.exclude.split(','))
    if name_shortener:
        data = dict((name_shortener(key), val) for (key, val) in data.items())
        if args.against is not None:
            args.against = [name_shortener(key) for key in args.against]
    lines = run_wilcoxon_test_on_data(data, args.against, args.threshold, args.raw)
    plots = create_scatterplots(data, args.against) if args.with_scatterplots else {}
    return lines, plots


def run_wilcoxon_test_on_data(data: Dict[str, Dict[str, Dict[str, float]]],
                              against: Optional[List[str]] = None, threshold: float = 0.05,
                              raw: bool = False) -> List[str]:
    """
    Performs all pairwise comparisons on the provided data, except if "against" was specified,
    compare every other run against the "against" run.
    :param data: scores such that data[run][structure][subject] = dice score
    :param against: runs to compare against; or None to compare all against all
    :param raw: whether to interpret Wilcoxon Z values "raw" or apply a correction
    :param threshold: p value to apply in deciding whether a result is significant
    """
    runs = sorted(data.keys())
    lines = []
    if against == []:
        against = None
    if against is not None:
        pairs = sorted([(run in against, run) for run in runs])
        runs = [pair[1] for pair in pairs]
    while runs:
        run1 = runs[0]
        runs = runs[1:]
        for run2 in runs:
            if against is not None and run2 not in against:
                continue
            results = evaluate_data_pair(data[run1], data[run2], raw)
            result_lines = compose_pairwise_result(threshold, results)
            if len(result_lines) > 0:
                lines.extend([f"Run 1: {run1}", f"Run 2: {run2}"])
                lines.extend(result_lines)
                lines.append("")
    return lines


def main() -> None:
    """
    Parse the arguments and run the Wilcoxon Signed-Rank test
    """
    lines, plots = wilcoxon_signed_rank_test(WilcoxonTestConfig.parse_args())
    for line in lines:
        print(line)
    for basename, fig in plots.items():
        fig.savefig(f"{basename}.png")


if __name__ == "__main__":
    main()

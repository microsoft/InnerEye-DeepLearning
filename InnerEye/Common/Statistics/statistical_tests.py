#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Some non-parametric statistical tests.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import math


def wilcoxon_z(vals1: Union[Dict[Any, float], List[float]],
               vals2: Optional[Union[Dict[Any, float], List[float]]] = None) -> float:
    """
    Applies the Wilcoxon signed-rank test, as for wilcoxon(vals1, vals2), but returns
    only the z score, not a triple (z, w, n).
    """
    return wilcoxon(vals1, vals2)[0]


def wilcoxon(vals1: Union[Dict[Any, float], List[float]],
             vals2: Optional[Union[Dict[Any, float], List[float]]] = None) -> Tuple[float, float, int]:
    """
    Applies the Wilcoxon signed-rank test, https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test .
    Given lists of numbers vals1 and vals2, which must be of the same length, returns a triple
    (z, w, n), where:
      n is the number of pairs (v1, v2) from corresponding elements of vals1 and vals2 such that
        v1 != v2
      w is the Wilcoxon W statistic for those pairs
      z is the z score for that W under the normal approximation, which is only accurate when n >= 10.
    Alternatively, vals2 can be None, in which case vals1 is interpreted as a list of differences;
    or vals1 and vals2 can both be dictionaries, in which case we pair on their shared keys.
    """
    if vals2 is None:
        abs_and_sign = sorted(absolute_and_sign(v) for v in vals1 if v != 0)
    elif isinstance(vals1, list) and isinstance(vals2, list):
        if len(vals1) != len(vals2):
            raise ValueError('Arguments must be of same length')
        abs_and_sign = sorted(absolute_and_sign(v2 - v1) for (v1, v2) in zip(vals1, vals2) if v1 != v2)
    else:
        # Assume both are dictionaries; mypy gets confused by types because this method is polymorphic.
        keys = [key for key in vals1 if key in vals2 and vals1[key] != vals2[key]]  # type: ignore
        abs_and_sign = sorted(absolute_and_sign(vals2[key] - vals1[key]) for key in keys  # type: ignore
                              if vals1[key] != vals2[key])  # type: ignore
    n = len(abs_and_sign)
    if n == 0:
        return (0, 0, 0)
    w = wilcoxon_w(abs_and_sign)
    var = n * (n + 1) * (2 * n + 1) / 6.0
    z = w / math.sqrt(var)
    return z, w, n


def absolute_and_sign(val: Union[float, int]) -> Tuple[Union[float, int], int]:
    """
    Returns a pair consisting of the absolute value of val and its sign.
    """
    if val > 0:
        return val, 1
    else:
        return -val, -1


def wilcoxon_w(pairs: List[Tuple[Union[float, int], int]]) -> float:
    """
    Returns the Wilcoxon "W" value, with rank smoothing; see Wikipedia article.
    """
    i = 0
    w = 0.0
    while i < len(pairs):
        absi = pairs[i][0]
        sum_sgn = pairs[i][1]
        j = i + 1
        while j < len(pairs) and pairs[j][0] == absi:
            sum_sgn += pairs[j][1]
            j += 1
        r = 1 + 0.5 * (i + j - 1)  # smoothed rank
        w += r * sum_sgn
        i = j
    return w


def mcnemar_z(vals1: Union[int, List[Union[float, int]]], vals2: Union[int, List[Union[float, int]]]) -> float:
    """
    Applies the McNemar test to the provided pairs of values, returning only the "z" value.
    """
    return mcnemar(vals1, vals2)[0]


def mcnemar(vals1: Union[int, List[Union[float, int]]], vals2: Union[int, List[Union[float, int]]]) -> \
        Tuple[float, float, int]:
    """
    Applies McNemar's test, https://en.wikipedia.org/wiki/McNemar%27s_test .
    Given lists of numbers vals1 and vals2, which must be of the same length, returns a triple
    (z, cs, n), where:
      n  is the number of pairs (v1, v2) from corresponding elements of vals1 and vals2 such
         that v1 != v2
      cs is the McNemar chi-squared statistic for the test (= z*z)
      z  is the signed z score for cs under the normal approximation.
    Alternatively, vals1 and vals2 can be integers, representing counts, and the test is applied
      directly to them.
    """
    n1, n2 = convert_pairs_to_comparison_counts(vals1, vals2)
    n = n1 + n2
    if n == 0:
        return 0, 0, 0
    z = (n2 - n1) / math.sqrt(n)
    cs = z * z
    return z, cs, n


def convert_pairs_to_comparison_counts(vals1: Union[int, List[float]], vals2: Union[int, List[float]]) \
        -> Tuple[int, int]:
    """
    If vals1 and vals2 are both ints, they are assumed already to be counts and are returned as is.
    Otherwise, they must be lists of the same length and are paired item by item. We return (n1, n2), where
    n1 is the number of pairs (val1, val2) from the two lists such that val1 < val2, and n2 is the number
    of pairs such that val1 > val2 (so equal pairs do not contribute to either count).
    """
    if isinstance(vals1, int) and isinstance(vals2, int):
        # The arguments are already counts.
        return vals1, vals2
    if isinstance(vals1, list) and isinstance(vals2, list) and len(vals1) == len(vals2):
        n1 = 0
        n2 = 0
        for (v1, v2) in zip(vals1, vals2):
            d = v2 - v1
            if d > 0:
                n2 += 1
            elif d < 0:
                n1 += 1
        return n1, n2
    raise ValueError('Arguments must both be ints or both be lists of the same length')

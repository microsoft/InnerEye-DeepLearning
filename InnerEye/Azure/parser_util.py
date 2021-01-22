#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from enum import Enum
from typing import Any


def _is_empty(item: Any) -> bool:
    """
    Returns True if the argument has length 0.
    :param item: Object to check.
    :return: True if the argument has length 0. False otherwise.
    """
    return hasattr(item, '__len__') and len(item) == 0


def _is_empty_or_empty_string_list(item: Any) -> bool:
    """
    Returns True if the argument has length 0, or a list with a single element that has length 0.
    :param item: Object to check.
    :return: True if argument has length 0, or a list with a single element that has length 0. False otherwise.
    """
    if _is_empty(item):
        return True
    if hasattr(item, '__len__') and len(item) == 1 and _is_empty(item[0]):
        return True
    return False


def value_to_string(x: object) -> str:
    """
    Returns a string representation of x, with special treatment of Enums (return their value)
    and lists (return comma-separated list).
    :param x: Object to convert to string
    :return: The string representation of the object.
    Special cases: For Enums, returns their value, for lists, returns a comma-separated list.
    """
    if isinstance(x, str):
        return x
    if isinstance(x, Enum):
        # noinspection PyUnresolvedReferences
        return x.value
    if isinstance(x, list):
        return ",".join(value_to_string(item) for item in x)
    return str(x)

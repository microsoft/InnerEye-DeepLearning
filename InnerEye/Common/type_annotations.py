#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar, Union

T = TypeVar('T')
PathOrString = Union[Path, str]
TupleStringOptionalFloat = Tuple[str, Optional[float]]
TupleInt2 = Tuple[int, int]
TupleInt3 = Tuple[int, int, int]
TupleFloat2 = Tuple[float, float]
TupleFloat3 = Tuple[float, float, float]
TupleFloat9 = Tuple[float, float, float, float, float, float, float, float, float]
IntOrTuple3 = Union[int, TupleInt3, Iterable]
DictStrFloat = Dict[str, float]
DictStrFloatOrFloatList = Dict[str, Union[float, List[float]]]

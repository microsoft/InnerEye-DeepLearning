#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
Alternative versions of tqdm.tqdm and tqdm.tqdm_notebook, so that we don't have to rely on the real tqdm.
This file must be located at top level so that batchflow's "import tqdm" works.
"""

from typing import Any


def tqdm(arg: Any, *_rest: Any) -> Any:
    return arg


def tqdm_notebook(arg: Any, *rest: Any) -> Any:
    return tqdm(arg, *rest)

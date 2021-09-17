#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from enum import Enum, unique


@unique
class SelectorTypes(Enum):
    """
    Contains the names of the columns in the CSV file that is written by model testing.
    """
    BaldSelector = "BaldSelector"
    PosteriorBasedSelector = "PosteriorBasedSelector"
    PosteriorBasedSelectorJoint = "PosteriorBasedSelectorJoint"
    GraphBasedSelector = "GraphBasedSelector"

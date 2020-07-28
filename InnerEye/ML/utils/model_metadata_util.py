#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import List

from InnerEye.Common.type_annotations import TupleInt3


def random_colour(rng: random.Random) -> TupleInt3:
    """
    Generates a random colour in RGB given a random number generator
    :param rng: Random number generator
    :return: Tuple with random colour in RGB
    """
    r = rng.randint(0, 255)
    g = rng.randint(0, 255)
    b = rng.randint(0, 255)
    return r, g, b


def generate_random_colours_list(rng: random.Random, size: int) -> List[TupleInt3]:
    """
    Generates a list of random colours in RGB given a random number generator and the size of this list
    :param rng: random number generator
    :param size: size of the list
    :return: list of random colours in RGB
    """
    return [random_colour(rng) for _ in range(size)]

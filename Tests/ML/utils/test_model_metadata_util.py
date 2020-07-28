#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random

from InnerEye.ML.utils.model_metadata_util import generate_random_colours_list, random_colour


def test_random_colour() -> None:
    """
    Test random colours
    """
    rng = random.Random(0)
    r, g, b = random_colour(rng)
    assert 0 <= r < 255
    assert 0 <= g < 255
    assert 0 <= b < 255


def test_generate_random_colours_list() -> None:
    """
    Test list of random colours
    """
    rng = random.Random(0)
    expected_list_len = 10
    list_colours = generate_random_colours_list(rng, 10)
    assert len(list_colours) == expected_list_len
    for r, g, b in list_colours:
        assert 0 <= r < 255
        assert 0 <= g < 255
        assert 0 <= b < 255

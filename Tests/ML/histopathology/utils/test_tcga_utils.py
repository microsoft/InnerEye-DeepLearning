#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd

from InnerEye.ML.Histopathology.utils.tcga_utils import extract_fields


def test_extract_fields() -> None:
    slide_id = "TCGA-XX-0123"
    tile_id = "ABCDEFGHIJKL"
    split = "train"
    label = 0
    path = (f"CRC_DX_{split.upper()}/"
            f"{['MSS', 'MSIMUT'][label]}/"
            f"blk-{tile_id}-{slide_id}-01Z-00-DX1.png")
    fields = {
        'slide_id': slide_id,
        'tile_id': tile_id,
        'image': path,
        'split': split,
        'label': label
    }
    extracted_fields = extract_fields(pd.Series(fields))
    assert fields == extracted_fields


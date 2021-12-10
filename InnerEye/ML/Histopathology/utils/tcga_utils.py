#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import pandas as pd


def extract_fields(row: pd.Series) -> dict:
    # Paths are structured as follows:
    # "CRC_DX_[TEST|TRAIN]/[MSS|MSIMUT]/blk-{tile_id}-{slide_id}-01Z-00-DX1.png"
    # - tile_id is an uppercase string of 12 letters
    # - slide_id is "TCGA-XX-XXXX"
    parts = row.image.split('/')
    return {
        'slide_id': parts[2][17:29],
        'tile_id': parts[2][4:16],
        'image': row.image,
        'label': {'MSS': 0, 'MSIMUT': 1}[parts[1]],
        'split': parts[0][7:].lower(),
    }

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd

from InnerEye.ML.utils.dataset_util import CategoricalToOneHotEncoder


def test_one_hot_encoder_with_infinite_values() -> None:
    df = pd.DataFrame(columns=["categorical"])
    df["categorical"] = ["F", "M", np.inf]
    encoder = CategoricalToOneHotEncoder.create_from_dataframe(df, ["categorical"])
    assert np.isnan(encoder.encode({"categorical": np.inf})).all()  # type: ignore

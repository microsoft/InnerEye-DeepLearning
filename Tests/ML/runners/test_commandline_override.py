#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse

from InnerEye.ML.model_config_base import ModelConfigBase
from Tests.ML.util import get_model_loader

MODEL_NAME = "DummyModelWithOverrideGroups"
LOADER = get_model_loader("Tests.ML.configs")


def test_script_params_override() -> None:
    # these are the parameters from the command line that should override
    # the initial parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_rate",
                        help="The name of the model to train/test.",
                        type=float,
                        default=1.0)
    args = parser.parse_args("")

    try:
        config: ModelConfigBase = LOADER.create_model_config_from_name(model_name=MODEL_NAME, overrides=vars(args))
        # check that the values were changed
        assert config.l_rate == args.l_rate
    except ValueError:
        # (Temporarily) handle the case where there is no Lung config.
        pass

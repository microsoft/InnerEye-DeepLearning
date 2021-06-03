#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging

from azureml.core import Model, Run

logging.getLogger().setLevel(logging.INFO)


def step4(input_model_folder: str, input_model_file: str) -> None:

    ws = Run.get_context().experiment.workspace

    model = Model.register(workspace=ws,
                           model_name='test_regression_model',
                           model_path=input_model_folder,
                           child_paths=[input_model_file],
                           model_framework=Model.Framework.PYTORCH,
                           model_framework_version="1.2",
                           description="test regression model")
    logging.info("Registered model id: %s, version: %s", model.id, model.version)


def main() -> None:
    logging.info("in main")

    parser = argparse.ArgumentParser("step4")
    parser.add_argument("--input_step4_model_folder", type=str, help="input_step4 model folder")
    parser.add_argument("--input_step4_model_file", type=str, help="input_step4 model file")

    args = parser.parse_args()
    logging.info("args: %s", args)

    step4(args.input_step4_model_folder,
          args.input_step4_model_file)


if __name__ == "__main__":
    logging.info("in wrapper")
    main()

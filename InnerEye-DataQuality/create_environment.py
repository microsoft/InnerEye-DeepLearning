#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import subprocess
from pathlib import Path

INNEREYE_DQ_ENVIRONMENT_FILE = Path(__file__).parent.absolute() / 'environment.yml'


def create_environment(environment_name: str = "InnerEyeDataQuality") -> None:
    print(f"Creating environment {environment_name} with the settings in "	
          f"{INNEREYE_DQ_ENVIRONMENT_FILE}")
    subprocess.Popen(
        f"conda env create --file {INNEREYE_DQ_ENVIRONMENT_FILE} --name {environment_name}",
        shell=True).communicate()


if __name__ == '__main__':
    create_environment()

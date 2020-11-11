#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

"""
IMPORTANT CONSTRAINTS
---------------------
DO NOT move this file, this file is expected to be in the root directory of this project by
the caller code in AML

This is a wrapper script that can be used by the ScriptRunConfig to launch python code that is baked in
the Docker image.  It can be used by either the Python or C# SDK to orchestrate code that is not managed
by Azure ML.

For example, if the python interpreter is located at /opt/miniconda/envs/azureml/bin/python in the Docker
container, and the python script is called /azureml/train.py,

runConfiguration.ScriptFile = new FileInfo(@"/Users/srmorin/train/python_wrapper.py");
runConfiguration.Arguments =
   new List<string> {"--spawnprocess", "/opt/miniconda/envs/azureml/bin/python", "/azureml/train.py"}
"""

import sys
from pathlib import Path

INNEREYE_SUBMODULE_NAME = "innereye-deeplearning"


def main():  # type: ignore
    project_root = Path(__file__).parent.absolute()
    submodule_dir = project_root / INNEREYE_SUBMODULE_NAME
    if submodule_dir.exists():
        sys.path += [str(submodule_dir)]
    import run_scoring
    run_scoring.run(project_root=project_root)


if __name__ == '__main__':
    main()

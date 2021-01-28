#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
from typing import Dict, List, Optional, Tuple


def spawn_and_monitor_subprocess(process: str, args: List[str], env: Optional[Dict[str, str]] = None) -> \
        Tuple[int, List[str]]:
    """
    Helper function to spawn and monitor subprocesses.
    :param process: The name or path of the process to spawn.
    :param args: The args to the process.
    :param env: The environment variables for the process (default is the environment variables of the parent).
    If not provided, copy the environment from the current process.
    :return: Return code after the process has finished, and the list of lines that were written to stdout by the
    subprocess.
    """
    if env is None:
        env = dict(os.environ.items())
    p = subprocess.Popen(
        [process] + args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True
    )

    # for mypy, we have just set stdout
    assert p.stdout

    # Read and print all the lines that are printed by the subprocess
    stdout_lines = []
    for line in iter(p.stdout.readline, ""):
        line = line.strip()
        stdout_lines.append(line)
        print(line)
    p.stdout.close()
    # return the subprocess error code to the calling job so that it is reported to AzureML
    return p.wait(), stdout_lines

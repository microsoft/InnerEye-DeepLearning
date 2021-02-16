#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
from typing import Dict, List, Optional, Tuple


def spawn_and_monitor_subprocess(process: str,
                                 args: List[str],
                                 env: Optional[Dict[str, str]] = None) -> \
        Tuple[int, List[str]]:
    """
    Helper function to start a subprocess, passing in a given set of arguments, and monitor it.
    Returns the subprocess exit code and the list of lines written to stdout.
    :param process: The name and path of the executable to spawn.
    :param args: The args to the process.
    :param env: The environment variables that the new process will run with. If not provided, copy the
    environment from the current process.
    :return: Return code after the process has finished, and the list of lines that were written to stdout by the
    subprocess.
    """
    if env is None:
        env = dict(os.environ.items())
    p = subprocess.Popen(
        args=[process] + args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True
    )

    # For mypy. We have set stdout in the arg list of Popen, so should be readable.
    assert p.stdout

    # Read and print all the lines that are printed by the subprocess
    stdout_lines = []
    for line in iter(p.stdout.readline, ""):
        line = line.strip()
        stdout_lines.append(line)
        print(line)
    p.stdout.close()
    return_code = p.wait()
    return return_code, stdout_lines

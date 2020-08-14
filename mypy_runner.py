#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def run_mypy(files: List[str]) -> int:
    """
    Runs mypy on the specified files, printing whatever is sent to stdout (i.e. mypy errors).
    Because of an apparent bug in mypy, we run mypy in --verbose mode, so that log lines are printed to
    stderr. We intercept these, and assume that any files mentioned in them have been processed.
    We run mypy repeatedly on the files that were not mentioned until there are none remaining, or until
    no further files are mentioned in the logs.
    :param files: list of .py files to check
    :return: maximum return code from any of the mypy runs
    """
    return_code = 0
    iteration = 1
    while files:
        print(f"Iteration {iteration}: running mypy on {len(files)}{' remaining' if iteration > 1 else ''} files")
        command = ["mypy", "--config=mypy.ini", "--verbose"] + files
        # Only stderr is piped, so stdout (containing the output we want) should be printed as normal.
        process = subprocess.run(command, stderr=subprocess.PIPE, text=True)
        # Set of files we are hoping to see mentioned in the mypy log.
        files_to_do = set(files)
        # Remove from files_to_do everything that's mentioned in the log.
        for line in process.stderr.split("\n"):
            for token in line.split():
                files_to_do.discard(token)
        # If we didn't manage to discard any files, there's no point continuing. This should not occur, but if
        # it does, we don't want to continue indefinitely.
        if len(files_to_do) == len(files):
            print("No further files appear to have been checked!")
            return_code = max(return_code, 1)
            break
        files = sorted(files_to_do)
        return_code = max(return_code, process.returncode)
        iteration += 1
    if return_code == 0:
        print("mypy SUCCEEDED")
    else:
        print("mypy FAILED")
    return return_code


def main() -> int:
    """
    Runs mypy on the files in the argument list, or every *.py file under the current directory if there are none.
    """
    exclude: List[str] = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if sys.argv[1:]:
        files = [Path(arg) for arg in sys.argv[1:]]
    else:
        files = sorted(map(lambda x: x.relative_to(current_dir), Path.cwd().rglob('*.py')))
    files = list(filter(lambda x: not any([str(Path(ele)) in str(x) for ele in exclude]), files))
    return run_mypy([str(file) for file in files])


if __name__ == "__main__":
    sys.exit(main())

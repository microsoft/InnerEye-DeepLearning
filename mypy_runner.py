#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path
from shutil import which
from typing import List


def run_mypy(files: List[str], mypy_executable_path: str) -> int:
    """
    Runs mypy on the specified files, printing whatever is sent to stdout (i.e. mypy errors).
    Because of an apparent bug in mypy, we run mypy in --verbose mode, so that log lines are printed to
    stderr. We intercept these, and assume that any files mentioned in them have been processed.
    We run mypy repeatedly on the files that were not mentioned until there are none remaining, or until
    no further files are mentioned in the logs.
    :param files: list of .py files to check
    :param mypy_executable_path: path to mypy executable
    :return: maximum return code from any of the mypy runs
    """
    return_code = 0
    print(f"Running mypy on {len(files)} files")
    for index, file in enumerate(files):
        print(f"Processing {(index+1):2d} of {len(files)}: {file}")
        file_path = Path(file)
        mypy_args = []
        if file_path.is_file():
            mypy_args = [file]
        elif file_path.is_dir():
            # There is a bug in recent mypy versions, complaining about duplicate files when telling
            # mypy to scan a directory. Telling it to scan a namespace avoids this bug.
            mypy_args = ["-p", file.replace(os.path.sep, ".")]
        else:
            print("Skipping.")
        if mypy_args:
            command = [mypy_executable_path, "--config=mypy.ini", *mypy_args]
            # We pipe stdout and then print it, otherwise lines can appear in the wrong order in builds.
            process = subprocess.run(command)
            return_code = max(return_code, process.returncode)
    return return_code


def main() -> int:
    """
    Runs mypy on the files in the argument list, or every *.py file under the current directory if there are none.
    """
    parser = ArgumentParser()
    parser.add_argument("-f", "--files", type=str, nargs='+', required=False, default=None,
                        help="List of files to run mypy on. If not provided, run on current directory")
    parser.add_argument("-m", "--mypy", type=str, required=False, default=None,
                        help="Path to mypy executable. If not provided, autodetect mypy executable.")
    args = parser.parse_args()
    current_dir = Path(".")
    if args.files:
        file_list = args.files
    else:
        file_list = list(str(f) for f in current_dir.glob('*.py'))
        for dir in ["InnerEye", "Tests", "TestsOutsidePackage", "TestSubmodule"]:
            file_list.append(dir)

    mypy = args.mypy or which("mypy")
    if not mypy:
        raise ValueError("Mypy executable not found.")

    return run_mypy(sorted(str(file) for file in file_list), mypy_executable_path=mypy)


if __name__ == "__main__":
    sys.exit(main())

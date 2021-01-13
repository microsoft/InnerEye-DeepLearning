#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import subprocess
import sys
from pathlib import Path
from typing import List
from shutil import which
from argparse import ArgumentParser


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
    iteration = 1
    while files:
        dirs = sorted(set(os.path.dirname(file) or "." for file in files))
        print(f"Iteration {iteration}: running mypy on {len(files)} files in {len(dirs)} directories")
        # Set of files we are hoping to see mentioned in the mypy log.
        files_to_do = set(files)
        for index, dir in enumerate(dirs, 1):
            # Adding "--no-site-packages" might be necessary if there are errors in site packages,
            # but it may stop inconsistencies with site packages being spotted.
            command = [mypy_executable_path, "--config=mypy.ini", "--verbose", dir]
            print(f"Processing directory {index:2d} of {len(dirs)}: {Path(dir).absolute()}")
            # We pipe stdout and then print it, otherwise lines can appear in the wrong order in builds.
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return_code = max(return_code, process.returncode)
            for line in process.stdout.split("\n"):
                if line and not line.startswith("Success: "):
                    tokens = line.split(":")
                    if line.startswith("Found") or len(tokens) < 2:
                        print(line)
                    else:
                        print(f"{Path.cwd() / tokens[0]}:{':'.join(tokens[1:])}")

            # Remove from files_to_do every Python file that's reported as processed in the log.
            for line in process.stderr.split("\n"):
                tokens = line.split()
                if len(tokens) == 4 and tokens[0] == "LOG:" and tokens[1] == "Parsing":
                    name = tokens[2]
                elif len(tokens) == 7 and tokens[:4] == ["LOG:", "Metadata", "fresh", "for"]:
                    name = tokens[-1]
                else:
                    continue
                if name.endswith(".py"):
                    if name.startswith("./") or name.startswith(".\\"):
                        name = name[2:]
                    files_to_do.discard(name)
        # If we didn't manage to discard any files, there's no point continuing. This should not occur, but if
        # it does, we don't want to continue indefinitely.
        if len(files_to_do) == len(files):
            print("No further files appear to have been checked! Unchecked files are:")
            for file in sorted(files_to_do):
                print(f"  {file}")
            return_code = max(return_code, 1)
            break
        files = sorted(files_to_do)
        iteration += 1
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
        file_list = [Path(arg) for arg in args.files if arg.endswith(".py")]
    else:
        # We don't want to check the files in the submodule if any, partly because they should already have
        # been checked in the original repo, and partly because we don't want the module name clashes mypy would
        # otherwise report.
        submodule_name = "innereye-deeplearning"
        files = set(current_dir.glob('*.py'))
        for path in current_dir.glob('*'):
            if path.name != submodule_name:
                files.update(path.rglob('*.py'))
        file_list = list(files)

    mypy = args.mypy or which("mypy")
    if not mypy:
        raise ValueError("Mypy executable not found.")

    return run_mypy(sorted(str(file) for file in file_list), mypy_executable_path=mypy)


if __name__ == "__main__":
    sys.exit(main())

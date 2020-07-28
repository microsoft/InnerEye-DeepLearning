#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Top level directory containing the InnerEye submodule; does not have to exist.
INNEREYE_SUBMODULE_NAME = "innereye-deeplearning"

PYTHONPATH_ENVIRONMENT_VARIABLE_NAME = "PYTHONPATH"


def spawn_and_monitor_subprocess(process: str, args: List[str], env: Dict[str, str]) -> int:
    """
    Helper function to spawn and monitor subprocesses.
    :param process: The name or path of the process to spawn.
    :param args: The args to the process.
    :param env: The environment variables for the process (default is the environment variables of the parent).
    :return: Return code after the process has finished.
    """
    p = subprocess.Popen(
        [process] + args,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )

    # Read and print all the lines that are printed by the subprocess
    for line in p.stdout:  # type: ignore
        print(line.decode('UTF-8').strip())

    # return the subprocess error code to the calling job so that it is reported to AzureML
    return p.wait()


def write_script(parser: argparse.ArgumentParser, script_path: Path, project_root: Path) -> None:
    """
    Writes a shell script based on the contents of the parsed arguments, ready to be run (under Python 3)
    as a subprocess.
    :param parser: argument parser
    :param script_path: path to write
    """
    args, unknown_args = parser.parse_known_args()
    if not unknown_args:
        raise ValueError("Expected a command starting with score.py or equivalent")

    with script_path.open(mode='w') as out:
        def run(line: str) -> None:
            out.write(f"{line.rstrip()}\n")

        def echo(line: str) -> None:
            out.write(f"echo {script_path}: {line}\n")
        # Set the PYTHONPATH to the project root and the InnerEye-DeepLearning submodule within it.
        # This ensures that the path is clean, there are no namespace clashes and that only the desired code is run.
        submodule_abs = project_root / INNEREYE_SUBMODULE_NAME
        # These lines can be uncommented for diagnostics:
        # echo(f"Listing {project_root}")
        # run(f"ls -o {project_root}")
        # echo(f"Listing {submodule_abs}")
        # run(f"ls -o {submodule_abs}")
        if submodule_abs.exists():
            pythonpath = f"{project_root}:{submodule_abs}"
        else:
            pythonpath = str(project_root)
        run(f"export {PYTHONPATH_ENVIRONMENT_VARIABLE_NAME}={pythonpath}")
        # We need to explicitly uninstall apex and radio because when given an explicit git+https URL,
        # pip does not check that the package is up to date, only that some version is already installed.
        echo("Uninstalling apex and radio")
        run(f"pip uninstall --yes apex radio")
        # Update the current conda environment (so no need to activate it afterwards, despite the message
        # given by conda). If we have an environment.yml file in both the top level and the submodule,
        # merge them.
        echo("Updating conda environment")
        top_level_env = Path("environment.yml")
        submodule_env = submodule_abs / "environment.yml"
        if not submodule_env.exists():
            merged_env = top_level_env
        elif top_level_env.exists():
            merged_env = Path("merged.yml")
            run("pip install conda-merge")
            run(f"conda-merge {submodule_env} {top_level_env} > {merged_env}")
        else:
            merged_env = submodule_env
        run(f"conda env update --name $CONDA_DEFAULT_ENV --file {merged_env}")
        # unknown_args should start with the script, so we prepend that with project_root if necessary.
        if not Path(unknown_args[0]).exists():
            unknown_args[0] = os.path.join(INNEREYE_SUBMODULE_NAME, unknown_args[0])
        # Now the environment should be suitable for actually running inference.
        echo(f"Starting scoring script {unknown_args[0]}")
        spawn_out = project_root / "spawn.out"
        spawn_err = project_root / "spawn.err"
        spawn_command = (f"{args.spawnprocess} {' '.join(unknown_args)} --data_root {args.data_folder} "
                         f"--project_root {project_root} " 
                         f"> {spawn_out} 2> {spawn_err}")
        echo(f"Command is: {spawn_command}")
        run(spawn_command)
        # Reinstate these for debugging if required
        # echo(f"Contents of {spawn_out}:")
        # run(f"cat {spawn_out}")
        # echo(f"Contents of {spawn_err}:")
        # run(f"cat {spawn_err}")
        echo("Finished")
    with script_path.open(mode='r') as inp:
        print(f"Contents of {script_path}:\n")
        print(inp.read())
        print("")
    return


def run(project_root: Path) -> None:
    parser = argparse.ArgumentParser(
        description='Execute code baked into a Docker Container from AzureML ScriptRunConfig')
    parser.add_argument('--spawnprocess', dest='spawnprocess', action='store', type=str)
    parser.add_argument('--data-folder', dest='data_folder', action='store', type=str)
    script_path = Path('run_score.sh')
    write_script(parser, script_path, project_root)
    code = spawn_and_monitor_subprocess(process='sh', args=[str(script_path)], env=dict(os.environ.items()))
    sys.exit(code)

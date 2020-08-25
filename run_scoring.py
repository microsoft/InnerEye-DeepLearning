#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from azureml.core import Model, Run

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
    :param project_root: main directory in which code may be found
    """
    args, unknown_args = parser.parse_known_args()
    if not unknown_args:
        raise ValueError("Expected a command starting with score.py or equivalent")

    with script_path.open(mode='w') as out:
        def run(line: str) -> None:
            out.write(f"{line.rstrip()}\n")

        def echo(line: str) -> None:
            out.write(f"echo {script_path}: {line}\n")
        # Set the PYTHONPATH to the project root and the InnerEye-DeepLearning submodule within it, and also
        # to any subdirectory (assumed to be a model directory) that contains either the submodule or "InnerEye".
        # This ensures that the path is clean, there are no namespace clashes and that only the desired code is run.
        submodule_abs = project_root / INNEREYE_SUBMODULE_NAME
        components = [project_root, submodule_abs]
        # These lines can be uncommented for diagnostics:
        # for c in components:
        #     echo(f"Listing path component {c}")
        #     run(f"ls -oR {c}")
        pythonpath = ":".join(str(c) for c in components if c.exists())
        run(f"export {PYTHONPATH_ENVIRONMENT_VARIABLE_NAME}={pythonpath}")
        if os.environ.get('CONDA_DEFAULT_ENV', None):
            # Environment may need updating (otherwise we assume it's been set at submission
            # time and is already correct).
            # We need to explicitly uninstall apex and radio because when given an explicit git+https URL,
            # pip does not check that the package is up to date, only that some version is already installed.
            echo("Uninstalling apex and radio")
            run("pip uninstall --yes apex radio")
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
        scoring_script = unknown_args[0]
        if not Path(scoring_script).exists():
            unknown_args[0] = os.path.join(INNEREYE_SUBMODULE_NAME, scoring_script)
        # Now the environment should be suitable for actually running inference.
        spawn_command = (f"{args.spawnprocess} {' '.join(unknown_args)} --data_root {args.data_folder} "
                         f"--project_root {project_root} ")
        run(spawn_command)
        echo("Finished")
    with script_path.open(mode='r') as inp:
        print(f"Contents of {script_path}:\n")
        print("=" * 80)
        print(inp.read())
        print("=" * 80)
    return


def run(project_root: Optional[Path] = None) -> None:
    """
    Runs inference on an image. This can be invoked in one of two ways:
    (1) when there is already a model in the project_root directory; this is the case when
    we arrive here from python_wrapper.py
    (2) when we need to download a model, which must be specified by the --model-id switch.
    This is the case when this script is invoked by submit_for_inference.py.
    :param project_root: the directory in which the model (including code) is located.
    Must be None if and only if the --model-id switch is provided.
    """
    parser = argparse.ArgumentParser(
        description='Execute code baked into a Docker Container from AzureML ScriptRunConfig')
    parser.add_argument('--spawnprocess', dest='spawnprocess', action='store', type=str)
    parser.add_argument('--data-folder', dest='data_folder', action='store', type=str)
    parser.add_argument('--model-id', dest='model_id', action='store', type=str)
    known_args, unknown_args = parser.parse_known_args()
    if known_args.model_id:
        if project_root:
            raise ValueError("--model-id should not be provided when project_root is specified")
        workspace = Run.get_context().experiment.workspace
        model = Model(workspace=workspace, id=known_args.model_id)
        current_dir = Path(".")
        project_root = Path(model.download(str(current_dir))).absolute()
    elif not project_root:
        raise ValueError("--model-id must be provided when project_root is unspecified")
    script_path = Path('run_score.sh')
    write_script(parser, script_path, project_root)
    print(f"Running {script_path} ...")
    env = dict(os.environ.items())
    # Work around https://github.com/pytorch/pytorch/issues/37377
    env['MKL_SERVICE_FORCE_INTEL'] = '1'
    code = spawn_and_monitor_subprocess(process='bash', args=[str(script_path)], env=env)
    sys.exit(code)


if __name__ == '__main__':
    run()

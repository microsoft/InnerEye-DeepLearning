#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Optional

import param
from azureml.core import Experiment
from tensorboard.program import TensorBoard

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import parse_arguments
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.generic_parsing import GenericConfig


class MonitorArguments(GenericConfig):
    """
    Stores all information that is need to start Tensorboard monitoring.
    """
    run_ids: Optional[str] = param.String(default=None,
                                          doc="A list of run ids to be monitored, separated by commas.")
    experiment_name: Optional[str] = param.String(default=None,
                                                  doc="The name of the experiment to monitor. This will fetch all "
                                                      "runs in the experiment.")
    run_status: str = param.String(default="Running,Completed",
                                   doc="A list of run status to filter the runs. Must be subset of "
                                       "[Running, Completed, Failed, Canceled]. Set to 'None' to not filter.")
    port: int = param.Integer(default=6006, bounds=(1, None), doc="Port to serve Tensorboard on. Default port is 6006")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    def validate(self) -> None:
        if not self.run_ids and not self.experiment_name:
            raise ValueError("You must provide either a list of run ids or an experiment name.")


def monitor(arguments: MonitorArguments, azure_config: AzureConfig) -> None:
    """
    Starts tensorboard monitoring as per the provided arguments.
    :param arguments: The arguments saying which runs should be monitored.
    :param azure_config: An AzureConfig object with secrets/keys to access the workspace.
    """
    # Fetch AzureML workspace and the experiment runs in it
    workspace = azure_config.get_workspace()

    if arguments.run_ids:
        run_ids = common_util.get_items_from_string(arguments.run_ids)
        if len(run_ids) == 0:
            print("At least one run_recovery_id must be given for monitoring.")
            exit(-1)
        exp_runs = [azure_util.fetch_run(workspace, run_id) for run_id in run_ids]
    else:
        if arguments.experiment_name not in workspace.experiments:
            print(
                f"The experiment: {arguments.experiment_name} doest not exist in the {arguments.workspace_name} "
                f"workspace.")
            exit(-1)

        experiment = Experiment(workspace, arguments.experiment_name)
        filters = common_util.get_items_from_string(arguments.run_status)

        exp_runs = azure_util.fetch_runs(experiment, filters)

        if len(exp_runs) == 0:
            print(f"No runs to monitor with status [{arguments.run_status}].")
            exit(-1)

    # Start Tensorboard on executing machine
    # ts = HotFixedTensorBoard(exp_runs, port=arguments.port)
    ts = TensorBoard(exp_runs, port=arguments.port)

    print("==============================================================================")
    for run in exp_runs:
        print(f"Run URL: {run.get_portal_url()}")
    print("Tensorboard URL: ")
    ts.start()
    print("==============================================================================\n\n")
    input("Press Enter to close Tensorboard...")
    ts.stop()


def parse_and_create_monitor() -> MonitorArguments:
    """
    Parses the given commandline arguments, and creates a class from them.
    :return: A MonitorArgument object that holds all information that is necessary to start monitoring.
    """
    parser = MonitorArguments.create_argparser()
    parser_result = parse_arguments(parser)
    return MonitorArguments(**parser_result.args)


def main(yaml_file_path: Path) -> None:
    """
    Parses the commandline arguments, and based on those, starts the Tensorboard monitoring for the AzureML runs
    supplied on the commandline.
    :param yaml_file_path: The path to the YAML config file that contains all Azure-related options (which workspace
    to access, etc)
    """
    arguments = parse_and_create_monitor()
    monitor(arguments, AzureConfig.from_yaml(yaml_file_path))


if __name__ == '__main__':
    main(fixed_paths.TRAIN_YAML_FILE)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any, List, Optional

import param
from azureml.core import Experiment
from azureml.tensorboard import Tensorboard

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.generic_parsing import GenericConfig


class AMLTensorBoardMonitorConfig(GenericConfig):
    """
    Stores all information that is need to start TensorBoard monitoring.
    """
    run_ids: Optional[List[str]] = param.List(class_=str, default=None, allow_None=True,
                                                doc="A list of run ids to be monitored.")
    experiment_name: Optional[str] = param.String(default=None, allow_None=True,
                                                  doc="The name of the experiment to monitor. This will fetch all "
                                                      "runs in the experiment.")
    local_root: Optional[Path] = param.ClassSelector(class_=Path, default=None, allow_None=True,
                                                     doc="An optional local directory to store the run logs in.")
    run_status: Optional[str] = param.String(default="Running,Completed",
                                             doc="A list of run status to filter the runs. Must be subset of "
                                                 "[Running, Completed, Failed, Canceled]. Set to 'None' to not filter.")
    port: int = param.Integer(default=6006, bounds=(1, None), doc="Port to serve TensorBoard on. Default port is 6006")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    def validate(self) -> None:
        if not self.run_ids and not self.experiment_name:
            raise ValueError("You must provide either a list of run ids or an experiment name.")


def monitor(monitor_config: AMLTensorBoardMonitorConfig, azure_config: AzureConfig) -> None:
    """
    Starts TensorBoard monitoring as per the provided arguments.
    :param monitor_config: The config containing information on which runs that need be monitored.
    :param azure_config: An AzureConfig object with secrets/keys to access the workspace.
    """
    # Fetch AzureML workspace and the experiment runs in it
    workspace = azure_config.get_workspace()

    if monitor_config.run_ids is not None:
        if len(monitor_config.run_ids) == 0:
            print("At least one run_recovery_id must be given for monitoring.")
            exit(-1)
        exp_runs = [azure_util.fetch_run(workspace, run_id) for run_id in monitor_config.run_ids]
    else:
        if monitor_config.experiment_name not in workspace.experiments:
            print(
                f"The experiment: {monitor_config.experiment_name} doesnt not "
                f"exist in the {monitor_config.workspace_name} workspace.")
            exit(-1)

        experiment = Experiment(workspace, monitor_config.experiment_name)
        filters = common_util.get_items_from_string(monitor_config.run_status) if monitor_config.run_status else []

        exp_runs = azure_util.fetch_runs(experiment, filters)

        if len(exp_runs) == 0:
            _msg = "No runs to monitor"
            if monitor_config.run_status:
                _msg += f"with status [{monitor_config.run_status}]."
            exit(-1)

    # Start TensorBoard on executing machine
    ts = Tensorboard(exp_runs, local_root=str(monitor_config.local_root), port=monitor_config.port)

    print("==============================================================================")
    for run in exp_runs:
        print(f"Run URL: {run.get_portal_url()}")
    print("TensorBoard URL: ")
    ts.start()
    print("==============================================================================\n\n")
    input("Press Enter to close TensorBoard...")
    ts.stop()


def main(yaml_file_path: Path) -> None:
    """
    Parses the commandline arguments, and based on those, starts the Tensorboard monitoring for the AzureML runs
    supplied on the commandline.
    :param yaml_file_path: The path to the YAML config file that contains all Azure-related options (which workspace
    to access, etc)
    """
    monitor(monitor_config=AMLTensorBoardMonitorConfig.parse_args(), azure_config=AzureConfig.from_yaml(yaml_file_path))


if __name__ == '__main__':
    main(fixed_paths.TRAIN_YAML_FILE)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os

from azureml._restclient.constants import RunStatus
from azureml.core import Experiment, Run, Workspace
from azureml.core.authentication import ServicePrincipalAuthentication


def cancel_running_and_queued_jobs() -> None:
    environ = os.environ
    auth = ServicePrincipalAuthentication(
        tenant_id='72f988bf-86f1-41af-91ab-2d7cd011db47',
        service_principal_id=environ["APPLICATION_ID"],
        service_principal_password=environ["APPLICATION_KEY"])
    workspace = Workspace.get(
        name="InnerEye-DeepLearning",
        auth=auth,
        subscription_id=environ["SUBSCRIPTION_ID"],
        resource_group="InnerEye-DeepLearning")
    branch = environ["BRANCH"]
    if not branch.startswith("refs/pull/"):
        print(f"This branch is not a PR branch, hence not cancelling anything: {branch}")
        exit(0)
    experiment_name = branch.replace("/", "_")
    experiment = Experiment(workspace, name=experiment_name)
    for run in experiment.get_runs(include_children=True, properties={}):
        assert isinstance(run, Run)
        if run.status in (RunStatus.QUEUED, RunStatus.RUNNING):
            print(f"Cancelling run {run.id} ({run.display_name})")
            # run.cancel()


if __name__ == "__main__":
    cancel_running_and_queued_jobs()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging

import azureml.core
from azureml.core import Workspace, Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

import pipeline


logging.getLogger().setLevel(logging.INFO)


def test_workspace() -> None:
    logging.info("SDK version: %s", azureml.core.VERSION)

    # from azureml.core.authentication import InteractiveLoginAuthentication
    # interactive_auth = InteractiveLoginAuthentication(tenant_id="d34bd10a-9e1d-417f-9290-6955ffb2a051")

    ws = Workspace.from_config()
    logging.info("name: %s, rg: %s, loc: %s, subs: %s", ws.name, ws.resource_group, ws.location, ws.subscription_id)

    datastore = ws.get_default_datastore()

    aml_compute_name = "cpu-cluster"
    try:
        aml_compute = AmlCompute(ws, aml_compute_name)
        logging.info("found existing compute target.")
    except ComputeTargetException:
        logging.info("creating new compute target")

        provisioning_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_D2_V2",
                                                                    min_nodes=0,
                                                                    max_nodes=1)

        aml_compute = ComputeTarget.create(ws, aml_compute_name, provisioning_config)
        aml_compute.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

    logging.info("Azure Machine Learning Compute attached: %s", aml_compute.get_status().serialize())

    pipeline.create_pipeline(ws, datastore, aml_compute, Experiment(ws, 'Hello_World1'))


def main() -> None:
    test_workspace()


if __name__ == "__main__":
    main()

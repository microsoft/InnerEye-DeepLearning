#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import getpass
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import param
from azureml.core import Keyvault, Run, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from azureml.train.estimator import Estimator
from azureml.train.hyperdrive import HyperDriveConfig

from InnerEye.Azure.azure_util import get_results_blob_path, get_run_id, \
    is_offline_run_context, to_azure_friendly_container_path
from InnerEye.Azure.secrets_handling import APPLICATION_KEY, SecretsHandling, read_variables_from_yaml
from InnerEye.Common import fixed_paths
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.utils.blobxfer_util import download_blobs


class VMPriority(Enum):
    """
    Configurations for VM priority to use for execution
    """
    LowPriority = 'lowpriority'
    Dedicated = 'dedicated'


# The name of the submit_to_azureml property of AzureConfig
AZURECONFIG_SUBMIT_TO_AZUREML = "submit_to_azureml"


class AzureConfig(GenericConfig):
    """
    Azure related configurations to set up valid workspace. Note that for a parameter to be settable (when not given
    on the command line) to a value from train_variables.yaml, its default here needs to be None and not the empty
    string, and its type will be Optional[str], not str.
    """
    subscription_id: str = param.String(None, doc="The subscription to use for AML jobs")
    tenant_id: str = param.String(None, doc="The tenant to use for AML jobs")
    application_id: str = param.String(None, doc="The application to use for AML jobs")
    storage_account: str = param.String(None, doc="The blob storage account to use to store outputs from AML jobs")
    datasets_storage_account: str = param.String(None,
                                                 doc="The blob storage account to use to access datasets in AML jobs")
    storage_account_secret_name: str = \
        param.String(None, doc="The name of the keyvault secret that contains the storage account key.")
    datasets_storage_account_secret_name: str = \
        param.String(None, doc="The name of the keyvault secret that contains the dataset storage account key.")
    datasets_container: str = param.String(None, doc="The blob storage container to use to access datasets in AML jobs")
    workspace_name: str = param.String(None, doc="The name of the AzureML workspace that should be used.")
    workspace_region: str = param.String(None, doc="The region to create AML workspaces in")
    resource_group: str = param.String(None, doc="The resource group to create AML workspaces in")
    docker_shm_size: str = param.String("440g", doc="The amount of memory available to experiments")
    node_count: int = param.Integer(1, bounds=(1, None), doc="Number of concurrent runs to launch")
    workers_per_node: int = param.Integer(1, bounds=(1, None), doc="Number of child runs for a run")
    hyperdrive: bool = param.Boolean(False, doc="Use HyperDrive for run execution")
    gpu_cluster_name: str = param.String(None, doc="GPU cluster to use if executing a run")
    pip_extra_index_url: Optional[str] = param.String(None, doc="An additional URL where PIP packages should be "
                                                                "loaded from.")
    submit_to_azureml: bool = param.Boolean(False, doc="If True, submit the executing script to run on AzureML.")
    is_train: bool = param.Boolean(True,
                                   doc="If True, train a new model. If False, run inference on an existing model.")
    model: str = param.String(doc="The name of the model to train/test.")
    register_model_only_for_epoch: Optional[int] = param.Integer(None,
                                                                 doc="If set, and run_recovery_id is also set, "
                                                                     "register the model for this epoch and do no "
                                                                     "training or testing")
    pytest_mark: Optional[str] = param.String(None,
                                              doc="If provided, run pytest after model training. pytest will only "
                                                  "run the tests that have the mark given in this argument "
                                                  "('--pytest_mark gpu' will run all tests marked with "
                                                  "'pytest.mark.gpu')")
    run_recovery_id: Optional[str] = param.String(None,
                                                  doc="A run recovery id string in the form 'experiment name:run id'"
                                                      " to use for inference or recovering a model training run.")
    build_number: int = param.Integer(0, doc="The numeric ID of the build that triggered this training run.")
    build_user: str = param.String(getpass.getuser(),
                                   doc="The user to associate this experiment with.")
    build_source_repository: str = param.String("InnerEye-DeepLearning",
                                                doc="The name of the repository this source belongs to.")
    build_branch: str = param.String(getpass.getuser() + "_local_branch",
                                     doc="The branch this experiment has been triggered from.")
    build_source_id: str = param.String("local_commit",
                                        doc="The git commit that was used to create this build.")
    build_source_message: str = param.String("Unknown",
                                             doc="The message associated with the git commit that was used to create "
                                                 "this build.")
    build_source_author: str = param.String("Unknown author",
                                            doc="The author of the git commit that was used to create this build.")
    user_friendly_name: Optional[str] = param.String(None, doc="A user friendly name to identify this experiment.")
    tag: Optional[str] = param.String(None, doc="A string that will be added as a tag to this experiment.")
    log_level: str = param.String("INFO",
                                  doc="The level of diagnostic information that should be printed out to the console.")
    wait_for_completion: bool = param.Boolean(False, doc="If true, wait until the AzureML job has completed or failed. "
                                                         "If false, submit and exit.")
    use_dataset_mount: bool = param.Boolean(False, doc="If true, consume an AzureML Dataset via mounting it "
                                                       "at job start. If false, consume it by downloading it at job "
                                                       "start. When running outside AzureML, datasets will always be "
                                                       "downloaded via blobxfer.")
    extra_code_directory: Optional[str] = param.String(None, doc="Directory (relative to project root) containing code "
                                                                 "(e.g. model config) to be included in the model for "
                                                                 "inference. Ignored by default.")
    project_root: Path = param.ClassSelector(class_=Path, default=fixed_paths.repository_root_directory(),
                                             doc="The root folder that contains all code of the project that starts "
                                                 "the InnerEye run.")
    _workspace: Workspace = param.ClassSelector(class_=Workspace,
                                                doc="The cached workspace object that has been created in the first"
                                                    "call to get_workspace")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)

    @staticmethod
    def from_yaml(yaml_file_path: Path) -> AzureConfig:
        """
        Creates an AzureConfig object with default values, with the keys/secrets populated from values in YAML files.
        :param yaml_file_path: path to the yaml to load as AzureConfig
        :return: AzureConfig with values populated from the yaml file.
        """
        return AzureConfig(**read_variables_from_yaml(yaml_file_path))

    def get_storage_account_key(self) -> str:
        """
        Gets the storage account key for the storage account that holds the AzureML run outputs.
        """
        return self.get_secret_from_keyvault(self.storage_account_secret_name)

    def get_dataset_storage_account_key(self) -> str:
        """
        Gets the storage account key for the storage account that holds the dataset.
        """
        return self.get_secret_from_keyvault(self.datasets_storage_account_secret_name)

    def get_secret_from_keyvault(self, secret_name: str) -> str:
        """
        Retrieves a secret from AzureML's workspace keyvault. If the secret is not found there, resort to reading from
        environment variables of the same name. Returns None if the secret was not found.
        :param secret_name: The name of the secret to retrieve.
        :return: The value of the secret, or None if it was not found in the keyvault or environment variable.
        """
        # Secrets can also be read from the run_context, but that is not available in an offline run.
        # Hence, creating a Keyvault instance from the workspace is easier.
        value = None
        try:
            value = Keyvault(self.get_workspace()).get_secret(secret_name)
        except:
            pass
        if value is None:
            raise ValueError(f"Unable to access secret '{secret_name}' in the workspace keyvault.")
        return value

    def get_workspace(self) -> Workspace:
        """
        Return a workspace object for an existing Azure Machine Learning Workspace (or default from YAML).
        When running inside AzureML, the workspace that is retrieved is always the one in the current
        run context. When running outside AzureML, it is created or accessed with the service principal.
        This function will read the workspace only in the first call to this method, subsequent calls will return
        a cached value.
        Throws an exception if the workspace doesn't exist or the required fields don't lead to a uniquely
        identifiable workspace.
        :return: Azure Machine Learning Workspace
        """
        if self._workspace:
            return self._workspace
        run_context = Run.get_context()
        if is_offline_run_context(run_context):
            service_principal_auth = self.get_service_principal_auth()
            self._workspace = Workspace.get(
                name=self.workspace_name,
                auth=service_principal_auth,
                subscription_id=self.subscription_id,
                resource_group=self.resource_group)
        else:
            self._workspace = run_context.experiment.workspace
        return self._workspace

    def get_service_principal_auth(self) -> Optional[Union[InteractiveLoginAuthentication,
                                                           ServicePrincipalAuthentication]]:
        """
        Creates a service principal authentication object with the application ID stored in the present object.
        The application key is read from the environment.
        :return: A ServicePrincipalAuthentication object that has the application ID and key or None if the key
         is not present
        """
        secrets_handler = SecretsHandling(project_root=self.project_root)
        application_key = secrets_handler.get_secret_from_environment(APPLICATION_KEY, allow_missing=True)
        if not application_key:
            logging.warning("Unable to retrieve the key for the Service Principal authentication "
                            f"(expected in environment variable '{APPLICATION_KEY}' or YAML). "
                            f"Switching to interactive login.")
            return InteractiveLoginAuthentication()

        return ServicePrincipalAuthentication(
            tenant_id=self.tenant_id,
            service_principal_id=self.application_id,
            service_principal_password=application_key)

    def download_outputs_from_run(self, blobs_path: Path,
                                  destination: Path,
                                  run: Optional[Run] = None,
                                  is_file: bool = False) -> Path:
        """
        Download the blobs from the run's storage container / DEFAULT_AML_UPLOAD_DIR.
        Silently returns for offline runs.
        :param blobs_path: Blobs path in DEFAULT_AML_UPLOAD_DIR of the run's storage container to download from
        :param run: Run to download from (default to current run if None)
        :param destination: Local path to save the downloaded blobs to
        :param is_file: Set to True if downloading a single file.
        :return: Destination root to the downloaded files
        """
        if self.storage_account is None:
            raise ValueError(f'self.storage_account cannot be None')
        key = self.get_storage_account_key()
        if key is None:
            raise ValueError(f'self.storage_account_key cannot be None')
        return download_blobs(
            account=self.storage_account,
            account_key=key,
            blobs_root_path=to_azure_friendly_container_path(
                Path(get_results_blob_path(get_run_id(run))) / fixed_paths.DEFAULT_AML_UPLOAD_DIR / blobs_path
            ),
            destination=destination,
            is_file=is_file
        )


@dataclass
class SourceConfig:
    """
    Contains all information that is required to submit a script to AzureML: Entry script, arguments,
    and information to set up the Python environment inside of the AzureML virtual machine.
    """
    root_folder: str
    entry_script: str
    conda_dependencies_files: List[Path]
    script_params: Optional[Dict[str, str]] = None
    hyperdrive_config_func: Optional[Callable[[Estimator], HyperDriveConfig]] = None
    upload_timeout_seconds: int = 36000
    environment_variables: Optional[Dict[str, str]] = None

    def set_script_params_except_submit_flag(self) -> None:
        """
        Populates the script_param field of the present object from the arguments in sys.argv, with the exception
        of the "submit_to_azureml" flag.
        """
        args = sys.argv[1:]
        submit_flag = f"--{AZURECONFIG_SUBMIT_TO_AZUREML}"
        retained_args = []
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith(submit_flag):
                if len(arg) == len(submit_flag):
                    # The argument list contains something like ["--submit_to_azureml", "True]: Skip 2 entries
                    i = i + 1
                elif arg[len(submit_flag)] != "=":
                    # The argument list contains a flag like "--submit_to_azureml_foo": Keep that.
                    retained_args.append(arg)
            else:
                retained_args.append(arg)
            i = i + 1
        # The AzureML documentation says that positional arguments should be passed in using an
        # empty string as the value.
        self.script_params = {arg: "" for arg in retained_args}


@dataclass
class ExperimentResultLocation:
    """
    Information that is need to recover where the results of an experiment reside.
    """
    results_container_name: Optional[str] = None
    results_uri: Optional[str] = None
    dataset_folder: Optional[str] = None
    dataset_uri: Optional[str] = None
    azure_job_name: Optional[str] = None
    commandline_overrides: Optional[str] = None


@dataclass
class ParserResult:
    """
    Stores the results of running an argument parser, broken down into a argument-to-value dictionary,
    arguments that the parser does not recognize, and settings that were read from YAML files.
    """
    args: Dict[str, Any]
    unknown: List[str]
    overrides: Dict[str, Any]
    known_settings_from_yaml: Dict[str, Any]
    unknown_settings_from_yaml: Dict[str, Any]

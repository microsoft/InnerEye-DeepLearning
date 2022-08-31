#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import getpass
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import param
from azureml.core import Run, ScriptRunConfig, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication, ServicePrincipalAuthentication
from azureml.train.hyperdrive import HyperDriveConfig
from git import Repo

from InnerEye.Azure.azure_util import fetch_run, is_offline_run_context
from InnerEye.Azure.secrets_handling import SecretsHandling, read_all_settings
from InnerEye.Common import fixed_paths
from InnerEye.Common.generic_parsing import GenericConfig

# The name of the "azureml" property of AzureConfig
AZURECONFIG_SUBMIT_TO_AZUREML = "azureml"


@dataclass(frozen=True)
class GitInformation:
    """
    Contains information about the git repository that was used to submit the present experiment.
    """
    repository: str
    branch: str
    commit_id: str
    commit_message: str
    commit_author: str
    is_dirty: bool


class AzureConfig(GenericConfig):
    """
    Azure related configurations to set up valid workspace. Note that for a parameter to be settable (when not given
    on the command line) to a value from settings.yml, its default here needs to be None and not the empty
    string, and its type will be Optional[str], not str.
    """
    subscription_id: str = param.String(doc="The ID of your Azure subscription.")
    tenant_id: str = param.String(doc="The Azure tenant ID.")
    application_id: str = param.String(doc="Optional: The ID of the Service Principal for authentication to Azure.")
    azureml_datastore: str = param.String(doc="The name of the AzureML datastore that holds the input training data. "
                                              "This must be created manually, and point to a folder inside the "
                                              "datasets storage account.")
    workspace_name: str = param.String(doc="The name of the AzureML workspace that should be used.")
    resource_group: str = param.String(doc="The Azure resource group that contains the AzureML workspace.")
    docker_shm_size: str = param.String("440g", doc="The shared memory in the docker image for the AzureML VMs.")
    hyperdrive: bool = param.Boolean(False, doc="If True, use AzureML HyperDrive for run execution.")
    cluster: str = param.String(doc="The name of the GPU cluster inside the AzureML workspace, that should "
                                    "execute the job.")
    pip_extra_index_url: str = \
        param.String(doc="An additional URL where PIP packages should be loaded from.")
    azureml: bool = param.Boolean(False, doc="If True, submit the executing script to run on AzureML.")
    tensorboard: bool = param.Boolean(False, doc="If True, then automatically launch TensorBoard to monitor the"
                                                 " latest submitted AzureML run.")
    train: bool = param.Boolean(True,
                                doc="If True, train a new model. If False, run inference on an existing model. For "
                                    "inference, you need to specify a --run_recovery_id=... as well.")
    model: str = param.String(doc="The name of the model to train/test.")
    only_register_model: bool = param.Boolean(False,
                                              doc="If set, and run_recovery_id is also set, register the model "
                                                  "that was trained in the recovery run, but don't do training"
                                                  "or inference.")
    pytest_mark: str = param.String(doc="If provided, run pytest instead of model training. pytest will only "
                                        "run the tests that have the mark given in this argument "
                                        "('--pytest_mark gpu' will run all tests marked with 'pytest.mark.gpu')")
    run_recovery_id: str = param.String(doc="A run recovery id string in the form 'experiment name:run id' "
                                            "to use for inference, recovering a model training run or to register "
                                            "a model.")
    experiment_name: str = param.String(doc="If provided, use this string as the name of the AzureML experiment. "
                                            "If not provided, create the experiment off the git branch name.")
    build_number: int = param.Integer(0, doc="The numeric ID of the Azure pipeline that triggered this training run.")
    build_user: str = param.String(getpass.getuser(),
                                   doc="The name of the user who started this run.")
    build_user_email: str = param.String(getpass.getuser(),
                                         doc="The email address of the user who started this run. Default: "
                                             "alias of the current user")
    build_source_repository: str = param.String(doc="The name of the repository this source belongs to.")
    build_branch: str = param.String(doc="The branch this experiment has been triggered from.")
    build_source_id: str = param.String(doc="The git commit that was used to create this build.")
    build_source_message: str = param.String(doc="The message associated with the git commit that was used to create "
                                                 "this build.")
    build_source_author: str = param.String(doc="The author of the git commit that was used to create this build.")
    tag: str = param.String(doc="A string that will be added as a tag to this experiment.")
    log_level: str = param.String("INFO",
                                  doc="The level of diagnostic information that should be printed out to the console.")
    wait_for_completion: bool = param.Boolean(False, doc="If true, wait until the AzureML job has completed or failed. "
                                                         "If false, submit and exit.")
    use_dataset_mount: bool = param.Boolean(False, doc="If true, consume an AzureML Dataset via mounting it "
                                                       "at job start. If false, consume it by downloading it at job "
                                                       "start. When running outside AzureML, datasets will always be "
                                                       "downloaded.")
    extra_code_directory: str = param.String(doc="Directory (relative to project root) containing code "
                                                 "(e.g. model config) to be included in the model for "
                                                 "inference. Ignored by default.")
    project_root: Path = param.ClassSelector(class_=Path, default=fixed_paths.repository_root_directory(),
                                             doc="The root folder that contains all code of the project "
                                                 "that starts the InnerEye run.")
    max_run_duration: str = param.String(doc="The maximum runtime that is allowed for this job when running in "
                                             "AzureML. This is a floating point number with a string suffix s, m, h, d "
                                             "for seconds, minutes, hours, day. Examples: '3.5h', '2d'")
    num_nodes: int = param.Integer(default=1, doc="The number of virtual machines that will be allocated for this"
                                                  "job in AzureML.")
    _workspace: Workspace = param.ClassSelector(class_=Workspace,
                                                doc="The cached workspace object that has been created in the first"
                                                    "call to get_workspace")

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.git_information: Optional[GitInformation] = None

    def validate(self) -> None:
        if self.only_register_model and not self.run_recovery_id:
            raise ValueError("If only_register_model is set, must also provide a valid run_recovery_id")

    def get_git_information(self) -> GitInformation:
        """
        Gets all version control information about the present source code in the project_root_directory.
        Information is taken from commandline arguments, or if not given there, retrieved from git directly.
        The result of the first call to this function is cached, and returned in later calls.
        """
        if self.git_information:
            return self.git_information
        branch = self.build_branch
        commit_id = self.build_source_id
        commit_author = self.build_source_author
        commit_message = self.build_source_message
        repository = self.build_source_repository or self.project_root.name
        is_dirty = True
        # noinspection PyBroadException
        try:
            logging.debug(f"Trying to read git repository on {self.project_root}")
            git_repo = Repo(self.project_root)
            try:
                active_branch = git_repo.active_branch.name
            except TypeError:
                # If the repository is in "detached head" state, getting the active branch fails.
                # In particular, this is the case on the build agents.
                active_branch = ""
            branch = branch or active_branch
            last_commit = git_repo.head.commit
            commit_id = commit_id or last_commit.hexsha
            commit_author = commit_author or last_commit.author.name
            commit_message = commit_message or last_commit.message[:120].strip()
            # Is_dirty in the present settings ignores untracked files.
            is_dirty = git_repo.is_dirty()
        except:
            logging.debug("This folder does not seem to be a git repository.")
        return GitInformation(
            repository=repository,
            branch=branch,
            commit_id=commit_id,
            commit_message=commit_message,
            commit_author=commit_author,
            is_dirty=is_dirty
        )

    @staticmethod
    def from_yaml(yaml_file_path: Path, project_root: Optional[Path]) -> AzureConfig:
        """
        Creates an AzureConfig object with default values, with the keys/secrets populated from values in the
         given YAML file. If a `project_root` folder is provided, a private settings file is read from there as well.

        :param yaml_file_path: Path to the YAML file that contains values to create the AzureConfig
        :param project_root: A folder in which to search for a private settings file.
        :return: AzureConfig with values populated from the yaml files.
        """
        config = AzureConfig(**read_all_settings(project_settings_file=yaml_file_path,
                                                 project_root=project_root))
        if project_root:
            config.project_root = project_root
        return config

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
            if self.subscription_id and self.resource_group:
                service_principal_auth = self.get_service_principal_auth()
                self._workspace = Workspace.get(
                    name=self.workspace_name,
                    auth=service_principal_auth,
                    subscription_id=self.subscription_id,
                    resource_group=self.resource_group)
            else:
                raise ValueError("The values for 'subscription_id' and 'resource_group' were not found. "
                                 "Was the Azure setup completed?")
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
        application_key = secrets_handler.get_secret_from_environment(fixed_paths.SERVICE_PRINCIPAL_KEY,
                                                                      allow_missing=True)
        if not application_key:
            logging.info("Using interactive login to Azure. To use Service Principal authentication, "
                         f"supply the password in in environment variable '{fixed_paths.SERVICE_PRINCIPAL_KEY}'.")
            return InteractiveLoginAuthentication()

        return ServicePrincipalAuthentication(
            tenant_id=self.tenant_id,
            service_principal_id=self.application_id,
            service_principal_password=application_key)

    def fetch_run(self, run_recovery_id: str) -> Run:
        """
        Gets an instantiated Run object for a given run recovery ID (format experiment_name:run_id).

        :param run_recovery_id: A run recovery ID (format experiment_name:run_id)
        """
        return fetch_run(workspace=self.get_workspace(), run_recovery_id=run_recovery_id)


@dataclass
class SourceConfig:
    """
    Contains all information that is required to submit a script to AzureML: Entry script, arguments,
    and information to set up the Python environment inside of the AzureML virtual machine.
    """
    root_folder: Path
    entry_script: Path
    conda_dependencies_files: List[Path]
    script_params: List[str] = field(default_factory=list)
    hyperdrive_config_func: Optional[Callable[[ScriptRunConfig], HyperDriveConfig]] = None
    upload_timeout_seconds: int = 36000
    environment_variables: Optional[Dict[str, str]] = None


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

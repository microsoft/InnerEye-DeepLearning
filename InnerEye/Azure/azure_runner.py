#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import getpass
import hashlib
import logging
import os
import signal
import sys
from argparse import ArgumentError, ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from azureml.core import Environment, Experiment, Run, ScriptRunConfig
from azureml.core.runconfig import MpiConfiguration, RunConfiguration
from azureml.core.workspace import WORKSPACE_DEFAULT_BLOB_STORE_NAME
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, ParserResult, SourceConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, RUN_RECOVERY_FROM_ID_KEY_NAME, \
    RUN_RECOVERY_ID_KEY_NAME, is_offline_run_context, merge_conda_dependencies
from InnerEye.Azure.secrets_handling import read_all_settings
from InnerEye.Azure.tensorboard_monitor import AMLTensorBoardMonitorConfig, monitor
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.utils.config_loader import ModelConfigLoader

SLEEP_TIME_SECONDS = 30

RUN_RECOVERY_FILE = "most_recent_run.txt"
# The version to use when creating an AzureML Python environment. We create all environments with a unique hashed
# name, hence version will always be fixed
ENVIRONMENT_VERSION = "1"

# Environment variables used for multi-node training
ENV_AZ_BATCHAI_MPI_MASTER_NODE = "AZ_BATCHAI_MPI_MASTER_NODE"
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_IP = "MASTER_IP"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"
ENV_NODE_RANK = "NODE_RANK"
ENV_GLOBAL_RANK = "GLOBAL_RANK"
ENV_LOCAL_RANK = "LOCAL_RANK"


def submit_to_azureml(azure_config: AzureConfig,
                      source_config: SourceConfig,
                      all_azure_dataset_ids: List[str],
                      all_dataset_mountpoints: List[str]) -> Run:
    """
    The main entry point when submitting the runner script to AzureML.
    It creates an AzureML workspace if needed, submits an experiment using the code
    as specified in source_config, and waits for completion if needed.
    :param azure_config: azure related configurations to setup valid workspace
    :param source_config: The information about which code should be submitted, and which arguments should be used.
    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using mounted datasets in AzureML, these are the per-dataset mount points.
    The list must have the same length as all_azure_dataset_ids.
    """
    azure_run: Optional[Run] = None

    # When running as part of the PR build, jobs frequently get interrupted by new pushes to the repository.
    # In this case, we'd like to cancel the current AzureML run before exiting, to reduce cost.
    # However, at present, this does NOT work, the SIGINT is not propagated through.
    def interrupt_handler(signal: int, _: Any) -> None:
        logging.info('Process interrupted via signal {}'.format(str(signal)))
        if azure_run:
            logging.info('Trying to terminate the AzureML job now.')
            azure_run.cancel()
        sys.exit(0)

    for s in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(s, interrupt_handler)
    # create train/test experiment
    script_run_config = create_run_config(azure_config, source_config, all_azure_dataset_ids, all_dataset_mountpoints)
    commandline_args = " ".join(source_config.script_params)
    azure_run = create_and_submit_experiment(azure_config, script_run_config, commandline_args=commandline_args)

    if azure_config.wait_for_completion:
        # We want the job output to be visible on the console, but the program should not exit if the
        # job fails because we need to download the pytest result file.
        azure_run.wait_for_completion(show_output=True, raise_on_error=False)

    return azure_run


def get_git_tags(azure_config: AzureConfig) -> Dict[str, str]:
    """
    Creates a dictionary with git-related information, like branch and commit ID. The dictionary key is a string
    that can be used as a tag on an AzureML run, the dictionary value is the git information. If git information
    is passed in via commandline arguments, those take precedence over information read out from the repository.
    :param azure_config: An AzureConfig object specifying git-related commandline args.
    :return: A dictionary mapping from tag name to git info.
    """
    git_information = azure_config.get_git_information()
    return {
        "source_repository": git_information.repository,
        "source_branch": git_information.branch,
        "source_id": git_information.commit_id,
        "source_dirty": str(git_information.is_dirty),
        "source_author": git_information.commit_author,
        "source_message": git_information.commit_message,
    }


def set_run_tags(run: Run, azure_config: AzureConfig, commandline_args: str) -> None:
    """
    Set metadata for the run
    :param run: Run to set metadata for.
    :param azure_config: The configurations for the present AzureML job
    :param commandline_args: A string that holds all commandline arguments that were used for the present run.
    """
    git_information = get_git_tags(azure_config)
    run.set_tags({
        "tag": azure_config.tag,
        "model_name": azure_config.model,
        "execution_mode": ModelExecutionMode.TRAIN.value if azure_config.train else ModelExecutionMode.TEST.value,
        RUN_RECOVERY_ID_KEY_NAME: azure_util.create_run_recovery_id(run=run),
        RUN_RECOVERY_FROM_ID_KEY_NAME: azure_config.run_recovery_id,
        "build_number": str(azure_config.build_number),
        "build_user": azure_config.build_user,
        "build_user_email": azure_config.build_user_email,
        **git_information,
        "commandline_args": commandline_args,
        CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: -1,
    })


def create_experiment_name(azure_config: AzureConfig) -> str:
    """
    Gets the name of the AzureML experiment. This is taken from the commandline, or from the git branch.
    :param azure_config: The object containing all Azure-related settings.
    :return: The name to use for the AzureML experiment.
    """
    if azure_config.experiment_name:
        return azure_config.experiment_name
    branch = azure_config.get_git_information().branch
    # If no branch information is found anywhere, create an experiment name that is the user alias and a timestamp
    # at monthly granularity, so that not too many runs accumulate in that experiment.
    return branch or getpass.getuser() + f"_local_branch_{date.today().strftime('%Y%m')}"


def create_and_submit_experiment(azure_config: AzureConfig,
                                 script_run_config: ScriptRunConfig,
                                 commandline_args: str) -> Run:
    """
    Creates an AzureML experiment in the workspace and submits it for execution.
    :param azure_config: azure related configurations to setup a valid workspace.
    :param script_run_config: The configuration for the script that should be run inside of AzureML.
    :param commandline_args: A string with all commandline arguments that were provided to the runner. These are only
    used to set a tag on the submitted AzureML run.
    :returns: Run object for the submitted AzureML run
    """
    workspace = azure_config.get_workspace()
    experiment_name = create_experiment_name(azure_config)
    exp = Experiment(workspace=workspace, name=azure_util.to_azure_friendly_string(experiment_name))

    # submit a training/testing run associated with the experiment
    run: Run = exp.submit(script_run_config)

    if is_offline_run_context(run):
        # This codepath will only be executed in unit tests, when exp.submit is mocked.
        return run

    # Set metadata for the run.
    set_run_tags(run, azure_config, commandline_args=commandline_args)

    print("\n==============================================================================")
    print(f"Successfully queued new run {run.id} in experiment: {exp.name}")

    if azure_config.run_recovery_id:
        print(f"\nRecovered from: {azure_config.run_recovery_id}")

    recovery_id = azure_util.create_run_recovery_id(run)
    recovery_file = Path(RUN_RECOVERY_FILE)
    if recovery_file.exists():
        recovery_file.unlink()
    recovery_file.write_text(recovery_id)

    print("Experiment URL: {}".format(exp.get_portal_url()))
    print("Run URL: {}".format(run.get_portal_url()))
    print("If this run fails, re-start runner.py and supply these additional arguments: "
          f"--run_recovery_id={recovery_id}")
    print(f"The run recovery ID has been written to this file: {recovery_file}")
    print("==============================================================================")
    if azure_config.tensorboard and azure_config.azureml:
        print("Starting TensorBoard now because you specified --tensorboard")
        monitor(monitor_config=AMLTensorBoardMonitorConfig(run_ids=[run.id]), azure_config=azure_config)
    else:
        print(f"To monitor this run locally using TensorBoard, run the script: "
              f"InnerEye/Azure/tensorboard_monitor.py --run_ids={run.id}")
        print("==============================================================================")
    return run


def get_or_create_python_environment(azure_config: AzureConfig,
                                     source_config: SourceConfig,
                                     environment_name: str = "",
                                     register_environment: bool = True) -> Environment:
    """
    Creates a description for the Python execution environment in AzureML, based on the Conda environment
    definition files that are specified in `source_config`. If such environment with this Conda environment already
    exists, it is retrieved, otherwise created afresh.
    :param azure_config: azure related configurations to use for model scale-out behaviour
    :param source_config: configurations for model execution, such as name and execution mode
    :param environment_name: If specified, try to retrieve the existing Python environment with this name. If that
    is not found, create one from the Conda files provided. This parameter is meant to be used when running
    inference for an existing model.
    :param register_environment: If True, the Python environment will be registered in the AzureML workspace. If
    False, it will only be created, but not registered. Use this for unit testing.
    """
    # Merge the project-specific dependencies with the packages that InnerEye itself needs. This should not be
    # necessary if the innereye package is installed. It is necessary when working with an outer project and
    # InnerEye as a git submodule and submitting jobs from the local machine.
    # In case of version conflicts, the package version in the outer project is given priority.
    conda_dependencies, merged_yaml = merge_conda_dependencies(source_config.conda_dependencies_files)  # type: ignore
    if azure_config.pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {azure_config.pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    env_variables = {
        "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": str(source_config.upload_timeout_seconds),
        # Occasionally uploading data during the run takes too long, and makes the job fail. Default is 300.
        "AZUREML_RUN_KILL_SIGNAL_TIMEOUT_SEC": "900",
        "MKL_SERVICE_FORCE_INTEL": "1",
        **(source_config.environment_variables or {})
    }
    base_image = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"
    # Create a name for the environment that will likely uniquely identify it. AzureML does hashing on top of that,
    # and will re-use existing environments even if they don't have the same name.
    # Hashing should include everything that can reasonably change. Rely on hashlib here, because the built-in
    # hash function gives different results for the same string in different python instances.
    hash_string = "\n".join([merged_yaml, azure_config.docker_shm_size, base_image, str(env_variables)])
    sha1 = hashlib.sha1(hash_string.encode("utf8"))
    overall_hash = sha1.hexdigest()[:32]
    unique_env_name = f"InnerEye-{overall_hash}"
    try:
        env_name_to_find = environment_name or unique_env_name
        env = Environment.get(azure_config.get_workspace(), name=env_name_to_find, version=ENVIRONMENT_VERSION)
        logging.info(f"Using existing Python environment '{env.name}'.")
        return env
    except Exception:
        logging.info(f"Python environment '{unique_env_name}' does not yet exist, creating and registering it.")
    env = Environment(name=unique_env_name)
    env.docker.enabled = True
    env.docker.shm_size = azure_config.docker_shm_size
    env.python.conda_dependencies = conda_dependencies
    env.docker.base_image = base_image
    env.environment_variables = env_variables
    if register_environment:
        env.register(azure_config.get_workspace())
    return env


def create_dataset_consumptions(azure_config: AzureConfig,
                                all_azure_dataset_ids: List[str],
                                all_dataset_mountpoints: List[str]) -> List[DatasetConsumptionConfig]:
    """
    Sets up all the dataset consumption objects for the datasets provided. Datasets that have an empty name will be
    skipped.
    :param azure_config: azure related configurations to use for model scale-out behaviour
    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using the datasets in AzureML, these are the per-dataset mount points.
    :return: A list of DatasetConsumptionConfig, in the same order as datasets were provided in all_azure_dataset_ids,
    omitting datasets with an empty name.
    """
    dataset_consumptions: List[DatasetConsumptionConfig] = []
    if len(all_dataset_mountpoints) > 0:
        if len(all_azure_dataset_ids) != len(all_dataset_mountpoints):
            raise ValueError(f"The number of dataset mount points ({len(all_dataset_mountpoints)}) "
                             f"must equal the number of Azure dataset IDs ({len(all_azure_dataset_ids)})")
    else:
        all_dataset_mountpoints = [""] * len(all_azure_dataset_ids)
    for i, (dataset_id, mount_point) in enumerate(zip(all_azure_dataset_ids, all_dataset_mountpoints)):
        if dataset_id:
            dataset_consumption = azure_config.get_dataset_consumption(dataset_id, i, mount_point)
            dataset_consumptions.append(dataset_consumption)
        elif mount_point:
            raise ValueError(f"Inconsistent setup: Dataset name at index {i} is empty, but a mount point has "
                             f"been provided ('{mount_point}')")
    return dataset_consumptions


def create_run_config(azure_config: AzureConfig,
                      source_config: SourceConfig,
                      all_azure_dataset_ids: List[str],
                      all_dataset_mountpoints: List[str],
                      environment_name: str = "") -> ScriptRunConfig:
    """
    Creates a configuration to run the InnerEye training script in AzureML.
    :param azure_config: azure related configurations to use for model scale-out behaviour
    :param source_config: configurations for model execution, such as name and execution mode
    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using the datasets in AzureML, these are the per-dataset mount points.
    :param environment_name: If specified, try to retrieve the existing Python environment with this name. If that
    is not found, create one from the Conda files provided in `source_config`. This parameter is meant to be used
    when running inference for an existing model.
    :return: The configured script run.
    """
    dataset_consumptions = create_dataset_consumptions(azure_config, all_azure_dataset_ids, all_dataset_mountpoints)
    # AzureML seems to sometimes expect the entry script path in Linux format, hence convert to posix path
    entry_script_relative_path = source_config.entry_script.relative_to(source_config.root_folder).as_posix()
    logging.info(f"Entry script {entry_script_relative_path} ({source_config.entry_script} relative to "
                 f"source directory {source_config.root_folder})")
    max_run_duration = None
    if azure_config.max_run_duration:
        max_run_duration = run_duration_string_to_seconds(azure_config.max_run_duration)
    workspace = azure_config.get_workspace()
    run_config = RunConfiguration(
        script=entry_script_relative_path,
        arguments=source_config.script_params,
    )
    run_config.environment = get_or_create_python_environment(azure_config, source_config,
                                                              environment_name=environment_name)
    run_config.target = azure_config.cluster
    run_config.max_run_duration_seconds = max_run_duration
    if azure_config.num_nodes > 1:
        distributed_job_config = MpiConfiguration(node_count=azure_config.num_nodes)
        run_config.mpi = distributed_job_config
        run_config.framework = "Python"
        run_config.communicator = "IntelMpi"
        run_config.node_count = distributed_job_config.node_count
    if len(dataset_consumptions) > 0:
        run_config.data = {dataset.name: dataset for dataset in dataset_consumptions}
    # Use blob storage for storing the source, rather than the FileShares section of the storage account.
    run_config.source_directory_data_store = workspace.datastores.get(WORKSPACE_DEFAULT_BLOB_STORE_NAME).name
    script_run_config = ScriptRunConfig(
        source_directory=str(source_config.root_folder),
        run_config=run_config,
    )
    if azure_config.hyperdrive:
        script_run_config = source_config.hyperdrive_config_func(script_run_config)  # type: ignore
    return script_run_config


def create_runner_parser(model_config_class: type = None) -> argparse.ArgumentParser:
    """
    Creates a commandline parser, that understands all necessary arguments for running a script in Azure,
    plus all arguments for the given class. The class must be a subclass of GenericConfig.
    :param model_config_class: A class that contains the model-specific parameters.
    :return: An instance of ArgumentParser.
    """
    parser = AzureConfig.create_argparser()
    ModelConfigLoader.add_args(parser)
    if model_config_class is not None:
        if not issubclass(model_config_class, GenericConfig):
            raise ValueError(f"The given class must be a subclass of GenericConfig, but got: {model_config_class}")
        model_config_class.add_args(parser)

    return parser


def parse_args_and_add_yaml_variables(parser: ArgumentParser,
                                      yaml_config_file: Optional[Path] = None,
                                      project_root: Optional[Path] = None,
                                      fail_on_unknown_args: bool = False) -> ParserResult:
    """
    Reads arguments from sys.argv, modifies them with secrets from local YAML files,
    and parses them using the given argument parser.
    :param project_root: The root folder for the whole project. Only used to access a private settings file.
    :param parser: The parser to use.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    :param fail_on_unknown_args: If True, raise an exception if the parser encounters an argument that it does not
    recognize. If False, unrecognized arguments will be ignored, and added to the "unknown" field of the parser result.
    :return: The parsed arguments, and overrides
    """
    settings_from_yaml = read_all_settings(yaml_config_file, project_root=project_root)
    return parse_arguments(parser,
                           settings_from_yaml=settings_from_yaml,
                           fail_on_unknown_args=fail_on_unknown_args)


def _create_default_namespace(parser: ArgumentParser) -> Namespace:
    """
    Creates an argparse Namespace with all parser-specific default values set.
    :param parser: The parser to work with.
    :return:
    """
    # This is copy/pasted from parser.parse_known_args
    namespace = Namespace()
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if not hasattr(namespace, action.dest):
                if action.default is not argparse.SUPPRESS:
                    setattr(namespace, action.dest, action.default)
    for dest in parser._defaults:
        if not hasattr(namespace, dest):
            setattr(namespace, dest, parser._defaults[dest])
    return namespace


def parse_arguments(parser: ArgumentParser,
                    settings_from_yaml: Optional[Dict[str, Any]] = None,
                    fail_on_unknown_args: bool = False,
                    args: List[str] = None) -> ParserResult:
    """
    Parses a list of commandline arguments with a given parser, and adds additional information read
    from YAML files. Returns results broken down into a full arguments dictionary, a dictionary of arguments
    that were set to non-default values, and unknown arguments.
    :param parser: The parser to use
    :param settings_from_yaml: A dictionary of settings read from a YAML config file.
    :param fail_on_unknown_args: If True, raise an exception if the parser encounters an argument that it does not
    recognize. If False, unrecognized arguments will be ignored, and added to the "unknown" field of the parser result.
    :param args: Arguments to parse. If not given, use those in sys.argv
    :return: The parsed arguments, and overrides
    """
    if args is None:
        args = sys.argv[1:]
    # The following code is a slightly modified version of what happens in parser.parse_known_args. This had to be
    # copied here because otherwise we would not be able to achieve the priority order that we desire.
    namespace = _create_default_namespace(parser)
    known_settings_from_yaml = dict()
    unknown_settings_from_yaml = dict()
    if settings_from_yaml:
        for key, setting_from_yaml in settings_from_yaml.items():
            if hasattr(namespace, key):
                known_settings_from_yaml[key] = setting_from_yaml
                setattr(namespace, key, setting_from_yaml)
            else:
                unknown_settings_from_yaml[key] = setting_from_yaml
    if len(unknown_settings_from_yaml) > 0 and fail_on_unknown_args:
        raise ValueError(f'Unknown settings from YAML: {unknown_settings_from_yaml}')
    try:
        namespace, unknown = parser._parse_known_args(args, namespace)
        if hasattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR):
            unknown.extend(getattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR))
            delattr(namespace, argparse._UNRECOGNIZED_ARGS_ATTR)
    except ArgumentError:
        parser.print_usage(sys.stderr)
        err = sys.exc_info()[1]
        parser._print_message(str(err), sys.stderr)
        raise
    # Parse the arguments a second time, without supplying defaults, to see which arguments actually differ
    # from defaults.
    namespace_without_defaults, _ = parser._parse_known_args(args, Namespace())
    parsed_args = vars(namespace).copy()
    overrides = vars(namespace_without_defaults).copy()
    if len(unknown) > 0 and fail_on_unknown_args:
        raise ValueError(f'Unknown arguments: {unknown}')
    return ParserResult(
        args=parsed_args,
        unknown=unknown,
        overrides=overrides,
        known_settings_from_yaml=known_settings_from_yaml,
        unknown_settings_from_yaml=unknown_settings_from_yaml
    )


def run_duration_string_to_seconds(s: str) -> Optional[int]:
    """
    Parse a string that represents a timespan, and returns it converted into seconds. The string is expected to be
    floating point number with a single character suffix s, m, h, d for seconds, minutes, hours, day.
    Examples: '3.5h', '2d'. If the argument is an empty string, None is returned.
    :param s: The string to parse.
    :return: The timespan represented in the string converted to seconds.
    """
    s = s.strip()
    if not s:
        return None
    suffix = s[-1]
    if suffix == "s":
        multiplier = 1
    elif suffix == "m":
        multiplier = 60
    elif suffix == "h":
        multiplier = 60 * 60
    elif suffix == "d":
        multiplier = 24 * 60 * 60
    else:
        raise ArgumentError("s", f"Invalid suffix: Must be one of 's', 'm', 'h', 'd', but got: {s}")  # type: ignore
    return int(float(s[:-1]) * multiplier)


def set_environment_variables_for_multi_node() -> None:
    """
    Sets the environment variables that PyTorch Lightning needs for multi-node training.
    """

    if ENV_AZ_BATCHAI_MPI_MASTER_NODE in os.environ:
        # For AML BATCHAI
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_AZ_BATCHAI_MPI_MASTER_NODE]
    elif ENV_MASTER_IP in os.environ:
        # AKS
        os.environ[ENV_MASTER_ADDR] = os.environ[ENV_MASTER_IP]
    else:
        logging.info("No settings for the MPI central node found. Assuming that this is a single node training job.")
        return

    if ENV_MASTER_PORT not in os.environ:
        os.environ[ENV_MASTER_PORT] = "6105"

    if ENV_OMPI_COMM_WORLD_RANK in os.environ:
        os.environ[ENV_NODE_RANK] = os.environ[ENV_OMPI_COMM_WORLD_RANK]  # node rank is the world_rank from mpi run
    env_vars = ", ".join(f"{var} = {os.environ[var]}" for var in [ENV_MASTER_ADDR, ENV_MASTER_PORT, ENV_NODE_RANK])
    print(f"Distributed training: {env_vars}")

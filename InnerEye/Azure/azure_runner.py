#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import getpass
import logging
import signal
import sys
from argparse import ArgumentError, ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from azureml.core import Dataset, Experiment, Run
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.datastore import Datastore
from azureml.core.workspace import WORKSPACE_DEFAULT_BLOB_STORE_NAME
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig
from azureml.train.dnn import PyTorch

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, ParserResult, SourceConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, RUN_RECOVERY_FROM_ID_KEY_NAME, \
    RUN_RECOVERY_ID_KEY_NAME, \
    merge_conda_dependencies
from InnerEye.Azure.secrets_handling import read_all_settings
from InnerEye.Azure.tensorboard_monitor import AMLTensorBoardMonitorConfig, monitor
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.utils.config_util import ModelConfigLoader

SLEEP_TIME_SECONDS = 30
INPUT_DATA_KEY = "input_data"

RUN_RECOVERY_FILE = "most_recent_run.txt"


def submit_to_azureml(azure_config: AzureConfig,
                      source_config: SourceConfig,
                      model_config_overrides: str,
                      azure_dataset_id: str) -> Run:
    """
    The main entry point. It creates an AzureML workspace if needed, submits an experiment using the code
    as specified in source_config, and waits for completion if needed.
    :param azure_config: azure related configurations to setup valid workspace
    :param source_config: The information about which code should be submitted, and which arguments should be used.
    :param model_config_overrides: A string that describes which model parameters were overwritten by commandline
     arguments in the present run. This is only used for diagnostic purposes (it is set as a Tag on the run).
    :param azure_dataset_id: The name of the dataset on blob storage to be used for this run.
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
    azure_run = create_and_submit_experiment(azure_config, source_config, model_config_overrides,
                                             azure_dataset_id)

    if azure_config.wait_for_completion:
        # We want the job output to be visible on the console, but the program should not exit if the
        # job fails because we need to download the pytest result file.
        azure_run.wait_for_completion(show_output=True, raise_on_error=False)

    return azure_run


def set_run_tags(run: Run, azure_config: AzureConfig, model_config_overrides: str) -> None:
    """
    Set metadata for the run
    :param run: Run to set metadata for.
    :param azure_config: The configurations for the present AzureML job
    :param model_config_overrides: A string that describes which model parameters were overwritten by commandline
     arguments in the present run.
    """
    git_information = azure_config.get_git_information()
    run.set_tags({
        "tag": azure_config.tag,
        "model_name": azure_config.model,
        "execution_mode": ModelExecutionMode.TRAIN.value if azure_config.train else ModelExecutionMode.TEST.value,
        RUN_RECOVERY_ID_KEY_NAME: azure_util.create_run_recovery_id(run=run),
        RUN_RECOVERY_FROM_ID_KEY_NAME: azure_config.run_recovery_id,
        "build_number": str(azure_config.build_number),
        "build_user": azure_config.build_user,
        "build_user_email": azure_config.build_user_email,
        "source_repository": git_information.repository,
        "source_branch": git_information.branch,
        "source_id": git_information.commit_id,
        "source_message": git_information.commit_message,
        "source_author": git_information.commit_author,
        "source_dirty": str(git_information.is_dirty),
        "overrides": model_config_overrides,
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


def create_and_submit_experiment(
        azure_config: AzureConfig,
        source_config: SourceConfig,
        model_config_overrides: str,
        azure_dataset_id: str) -> Run:
    """
    Creates an AzureML experiment in the workspace and submits it for execution.
    :param azure_config: azure related configurations to setup valid workspace
    :param source_config: The information about which code should be submitted, and which arguments should be used.
    :param model_config_overrides: A string that describes which model parameters were overwritten by commandline
     arguments in the present run. This is only used for diagnostic purposes (it is set as a Tag on the run).
    :param azure_dataset_id: The name of the dataset in blob storage to be used for this run.
    :returns: Run object for the submitted AzureML run
    """
    workspace = azure_config.get_workspace()
    experiment_name = create_experiment_name(azure_config)
    exp = Experiment(workspace=workspace, name=azure_util.to_azure_friendly_string(experiment_name))
    pt_env = create_pytorch_environment(azure_config, source_config, azure_dataset_id)

    # submit a training/testing run associated with the experiment
    run: Run = exp.submit(pt_env)

    # set metadata for the run
    set_run_tags(run, azure_config, model_config_overrides)

    print("\nSuccessfully queued new run for experiment: {}".format(exp.name))
    print("==============================================================================")

    if azure_config.run_recovery_id:
        print(f"\nRecovered from: {azure_config.run_recovery_id}")

    recovery_id = azure_util.create_run_recovery_id(run)
    recovery_file = Path(RUN_RECOVERY_FILE)
    if recovery_file.exists():
        recovery_file.unlink()
    recovery_file.write_text(recovery_id)

    print("==============================================================================")
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


def get_or_create_dataset(azure_config: AzureConfig,
                          azure_dataset_id: str) -> Dataset:
    """
    Looks in the AzureML datastore for a dataset of the given name. If there is no such dataset, a dataset is created
    and registered, assuming that the files are in a folder that has the same name as the dataset. For example, if
    azure_dataset_id is 'foo', then the 'foo' dataset is pointing to <container_root>/datasets/foo folder.

    WARNING: the behaviour of Dataset.File.from_files, used below, is idiosyncratic. For example,
    if "mydataset" storage has two "foo..." subdirectories each containing
    a file dataset.csv and a directory ABC,

    datastore = Datastore.get(workspace, "mydataset")
    # This dataset has the file(s) in foo-bar01 at top level, e.g. dataset.csv
    ds1 = Dataset.File.from_files([(datastore, "foo-bar01/*")])
    # This dataset has two directories at top level, each with a name matching foo-bar*, and each
    # containing dataset.csv.
    ds2 = Dataset.File.from_files([(datastore, "foo-bar*/*")])
    # This dataset contains a single directory "mydataset" at top level, containing a subdirectory
    # foo-bar01, containing dataset.csv and (part of) ABC.
    ds3 = Dataset.File.from_files([(datastore, "foo-bar01/*"),
                                   (datastore, "foo-bar01/ABC/abc_files/*/*.nii.gz")])

    These behaviours can be verified by calling "ds.download()" on each dataset ds.
    """
    logging.info(f"Retrieving datastore '{azure_config.azureml_datastore}' from AzureML workspace")
    workspace = azure_config.get_workspace()
    datastore = Datastore.get(workspace, azure_config.azureml_datastore)
    try:
        logging.info(f"Trying to retrieve AzureML Dataset '{azure_dataset_id}'")
        azureml_dataset = Dataset.get_by_name(workspace, name=azure_dataset_id)
        logging.info("Dataset found.")
    except:
        logging.info(f"Dataset does not yet exist, creating a new one from data in folder '{azure_dataset_id}'")
        # See WARNING above before changing the from_files call!
        azureml_dataset = Dataset.File.from_files([(datastore, azure_dataset_id)])
        logging.info("Registering the dataset for future use.")
        azureml_dataset.register(workspace, name=azure_dataset_id)
    return azureml_dataset


def create_pytorch_environment(azure_config: AzureConfig,
                               source_config: SourceConfig,
                               azure_dataset_id: str) -> PyTorch:
    """
    Creates an Estimator environment required for model execution

    :param workspace: The AzureML workspace
    :param azure_config: azure related configurations to use for model scaleout behaviour
    :param source_config: configurations for model execution, such as name and execution mode
    :param azure_dataset_id: The name of the dataset in blob storage to be used for this run.
    :return: The configured PyTorch environment to be used for experimentation
    """
    azureml_dataset = get_or_create_dataset(azure_config, azure_dataset_id=azure_dataset_id)
    if azureml_dataset:
        if azure_config.use_dataset_mount:
            logging.info("Inside AzureML, the dataset will be provided as a mounted folder.")
            estimator_inputs = [azureml_dataset.as_named_input(INPUT_DATA_KEY).as_mount()]
        else:
            logging.info("Inside AzureML, the dataset will be downloaded before training starts.")
            estimator_inputs = [azureml_dataset.as_named_input(INPUT_DATA_KEY).as_download()]
    else:
        raise ValueError("No AzureML dataset was found.")

    return create_estimator_from_configs(azure_config, source_config, estimator_inputs)


def pytorch_version_from_conda_dependencies(conda_dependencies: CondaDependencies) -> Optional[str]:
    """
    Given a CondaDependencies object, look for a spec of the form "pytorch=...", and return
    whichever supported version is compatible with the value, or None if there isn't one.
    """
    supported_versions = PyTorch.get_supported_versions()
    for spec in conda_dependencies.conda_packages:
        components = spec.split("=")
        if len(components) == 2 and components[0] == "pytorch":
            version = components[1]
            for supported in supported_versions:
                if version.startswith(supported) or supported.startswith(version):
                    return supported
    return None


def create_estimator_from_configs(azure_config: AzureConfig,
                                  source_config: SourceConfig,
                                  estimator_inputs: List[DatasetConsumptionConfig]) -> PyTorch:
    """
    Create an return a PyTorch estimator from the provided configuration information.
    :param azure_config: Azure configuration, used to store various values for the job to be submitted
    :param source_config: source configutation, for other needed values
    :param estimator_inputs: value for the "inputs" field of the estimator.
    :return:
    """
    # AzureML seems to sometimes expect the entry script path in Linux format, hence convert to posix path
    entry_script_relative_path = source_config.entry_script.relative_to(source_config.root_folder).as_posix()
    logging.info(f"Entry script {entry_script_relative_path} ({source_config.entry_script} relative to "
                 f"source directory {source_config.root_folder})")
    environment_variables = {
        "AZUREML_OUTPUT_UPLOAD_TIMEOUT_SEC": str(source_config.upload_timeout_seconds),
        "MKL_SERVICE_FORCE_INTEL": "1",
        **(source_config.environment_variables or {})
    }
    # Merge the project-specific dependencies with the packages that InnerEye itself needs. This should not be
    # necessary if the innereye package is installed. It is necessary when working with an outer project and
    # InnerEye as a git submodule and submitting jobs from the local machine.
    # In case of version conflicts, the package version in the outer project is given priority.
    conda_dependencies = merge_conda_dependencies(source_config.conda_dependencies_files)  # type: ignore
    if azure_config.pip_extra_index_url:
        # When an extra-index-url is supplied, swap the order in which packages are searched for.
        # This is necessary if we need to consume packages from extra-index that clash with names of packages on
        # pypi
        conda_dependencies.set_pip_option(f"--index-url {azure_config.pip_extra_index_url}")
        conda_dependencies.set_pip_option("--extra-index-url https://pypi.org/simple")
    # create Estimator environment
    framework_version = pytorch_version_from_conda_dependencies(conda_dependencies)
    assert framework_version is not None, "The AzureML SDK is behind PyTorch, it does not yet know the version we use."
    logging.info(f"PyTorch framework version: {framework_version}")
    max_run_duration = None
    if azure_config.max_run_duration:
        max_run_duration = run_duration_string_to_seconds(azure_config.max_run_duration)
    workspace = azure_config.get_workspace()
    estimator = PyTorch(
        source_directory=str(source_config.root_folder),
        entry_script=entry_script_relative_path,
        script_params=source_config.script_params,
        compute_target=azure_config.cluster,
        # Use blob storage for storing the source, rather than the FileShares section of the storage account.
        source_directory_data_store=workspace.datastores.get(WORKSPACE_DEFAULT_BLOB_STORE_NAME),
        inputs=estimator_inputs,
        environment_variables=environment_variables,
        shm_size=azure_config.docker_shm_size,
        use_docker=True,
        use_gpu=True,
        framework_version=framework_version,
        max_run_duration_seconds=max_run_duration
    )
    estimator.run_config.environment.python.conda_dependencies = conda_dependencies
    # We'd like to log the estimator config, but conversion to string fails when the Estimator has some inputs.
    # logging.info(azure_util.estimator_to_string(estimator))
    if azure_config.hyperdrive:
        estimator = source_config.hyperdrive_config_func(estimator)  # type: ignore
    return estimator


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
                                      fail_on_unknown_args: bool = False,
                                      args: List[str] = None) -> ParserResult:
    """
    Reads arguments from sys.argv, modifies them with secrets from local YAML files,
    and parses them using the given argument parser.
    :param project_root: The root folder for the whole project. Only used to access a private settings file.
    :param parser: The parser to use.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    :param fail_on_unknown_args: If True, raise an exception if the parser encounters an argument that it does not
    recognize. If False, unrecognized arguments will be ignored, and added to the "unknown" field of the parser result.
    :param args: arguments to parse
    :return: The parsed arguments, and overrides
    """
    settings_from_yaml = read_all_settings(yaml_config_file, project_root=project_root)
    return parse_arguments(parser,
                           settings_from_yaml=settings_from_yaml,
                           fail_on_unknown_args=fail_on_unknown_args,
                           args=args)


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
        raise ArgumentError("s", f"Invalid suffix: Must be one of 's', 'm', 'h', 'd', but got: {s}")
    return int(float(s[:-1]) * multiplier)

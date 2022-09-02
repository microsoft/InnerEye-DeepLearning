#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import getpass
import logging
import os
import sys
from argparse import ArgumentError, ArgumentParser, Namespace
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from health_azure import DatasetConfig

from InnerEye.Azure.azure_config import AzureConfig, ParserResult
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, RUN_RECOVERY_FROM_ID_KEY_NAME
from InnerEye.Azure.secrets_handling import read_all_settings
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.utils.config_loader import ModelConfigLoader

DEFAULT_DOCKER_BASE_IMAGE = "mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.2-cudnn8-ubuntu18.04"

# Environment variables used for multi-node training
ENV_AZ_BATCHAI_MPI_MASTER_NODE = "AZ_BATCHAI_MPI_MASTER_NODE"
ENV_AZ_BATCH_MASTER_NODE = "AZ_BATCH_MASTER_NODE"
ENV_MASTER_ADDR = "MASTER_ADDR"
ENV_MASTER_IP = "MASTER_IP"
ENV_MASTER_PORT = "MASTER_PORT"
ENV_OMPI_COMM_WORLD_RANK = "OMPI_COMM_WORLD_RANK"
ENV_NODE_RANK = "NODE_RANK"
ENV_GLOBAL_RANK = "GLOBAL_RANK"
ENV_LOCAL_RANK = "LOCAL_RANK"
MASTER_PORT_DEFAULT = 6105


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


def additional_run_tags(azure_config: AzureConfig, commandline_args: str) -> Dict[str, str]:
    """
    Gets the set of tags that will be added to the AzureML run as metadata, like git status and user name.

    :param azure_config: The configurations for the present AzureML job
    :param commandline_args: A string that holds all commandline arguments that were used for the present run.
    """
    git_information = get_git_tags(azure_config)
    return {
        "tag": azure_config.tag,
        "model_name": azure_config.model,
        "execution_mode": ModelExecutionMode.TRAIN.value if azure_config.train else ModelExecutionMode.TEST.value,
        RUN_RECOVERY_FROM_ID_KEY_NAME: azure_config.run_recovery_id,
        "build_number": str(azure_config.build_number),
        "build_user": azure_config.build_user,
        "build_user_email": azure_config.build_user_email,
        **git_information,
        "commandline_args": commandline_args,
        CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY: "-1",
    }


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


def create_dataset_configs(azure_config: AzureConfig,
                           all_azure_dataset_ids: List[str],
                           all_dataset_mountpoints: List[str],
                           all_local_datasets: List[Optional[Path]]) -> List[DatasetConfig]:
    """
    Sets up all the dataset consumption objects for the datasets provided. The returned list will have the same length
    as there are non-empty azure dataset IDs.

    Valid arguments combinations:
    N azure datasets, 0 or N mount points, 0 or N local datasets

    :param azure_config: azure related configurations to use for model scale-out behaviour
    :param all_azure_dataset_ids: The name of all datasets on blob storage that will be used for this run.
    :param all_dataset_mountpoints: When using the datasets in AzureML, these are the per-dataset mount points.
    :param all_local_datasets: The paths for all local versions of the datasets.
    :return: A list of DatasetConfig objects, in the same order as datasets were provided in all_azure_dataset_ids,
        omitting datasets with an empty name.
    """
    datasets: List[DatasetConfig] = []
    num_local = len(all_local_datasets)
    num_azure = len(all_azure_dataset_ids)
    num_mount = len(all_dataset_mountpoints)
    if num_azure > 0 and (num_local == 0 or num_local == num_azure) and (num_mount == 0 or num_mount == num_azure):
        # Test for valid settings: If we have N azure datasets, the local datasets and mount points need to either
        # have exactly the same length, or 0. In the latter case, empty mount points and no local dataset will be
        # assumed below.
        count = num_azure
    elif num_azure == 0 and num_mount == 0:
        # No datasets in Azure at all: This is possible for runs that for example download their own data from the web.
        # There can be any number of local datasets, but we are not checking that. In MLRunner.setup, there is a check
        # that leaves local datasets intact if there are no Azure datasets.
        return []
    else:
        raise ValueError("Invalid dataset setup. You need to specify N entries in azure_datasets and a matching "
                         "number of local_datasets and dataset_mountpoints")
    for i in range(count):
        azure_dataset = all_azure_dataset_ids[i] if i < num_azure else ""
        if not azure_dataset:
            continue
        mount_point = all_dataset_mountpoints[i] if i < num_mount else ""
        local_dataset = all_local_datasets[i] if i < num_local else None
        is_empty_azure_dataset = len(azure_dataset.strip()) == 0
        config = DatasetConfig(name=azure_dataset,
                               # Workaround for a bug in hi-ml 0.1.11: mount_point=="" creates invalid jobs,
                               # setting to None works.
                               target_folder=mount_point or None,
                               local_folder=local_dataset,
                               use_mounting=azure_config.use_dataset_mount,
                               datastore=azure_config.azureml_datastore)
        if is_empty_azure_dataset:
            config.name = ""
        datasets.append(config)
    return datasets


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
    if ENV_AZ_BATCH_MASTER_NODE in os.environ:
        master_node = os.environ[ENV_AZ_BATCH_MASTER_NODE]
        logging.debug(
            f"Found AZ_BATCH_MASTER_NODE: {master_node} in environment variables")
        # For AML BATCHAI
        split_master_node_addr = master_node.split(":")
        if len(split_master_node_addr) == 2:
            master_addr, port = split_master_node_addr
            os.environ[ENV_MASTER_PORT] = port
        elif len(split_master_node_addr) == 1:
            master_addr = split_master_node_addr[0]
        else:
            raise ValueError(f"Format not recognized: {master_node}")
        os.environ[ENV_MASTER_ADDR] = master_addr
    elif ENV_AZ_BATCHAI_MPI_MASTER_NODE in os.environ and os.environ.get(ENV_AZ_BATCHAI_MPI_MASTER_NODE) != "localhost":
        mpi_master_node = os.environ[ENV_AZ_BATCHAI_MPI_MASTER_NODE]
        logging.debug(
            f"Found AZ_BATCHAI_MPI_MASTER_NODE: {mpi_master_node} in environment variables")
        # For AML BATCHAI
        os.environ[ENV_MASTER_ADDR] = mpi_master_node
    elif ENV_MASTER_IP in os.environ:
        master_ip = os.environ[ENV_MASTER_IP]
        logging.debug(
            f"Found MASTER_IP: {master_ip} in environment variables")
        # AKS
        os.environ[ENV_MASTER_ADDR] = master_ip
    else:
        logging.info("No settings for the MPI central node found. Assuming that this is a single node training job.")
        return

    if ENV_MASTER_PORT not in os.environ:
        os.environ[ENV_MASTER_PORT] = str(MASTER_PORT_DEFAULT)

    if ENV_OMPI_COMM_WORLD_RANK in os.environ:
        world_rank = os.environ[ENV_OMPI_COMM_WORLD_RANK]
        logging.debug(f"Found OMPI_COMM_WORLD_RANK: {world_rank} in environment variables")
        os.environ[ENV_NODE_RANK] = world_rank  # node rank is the world_rank from mpi run

    env_vars = ", ".join(f"{var} = {os.environ[var]}" for var in [ENV_MASTER_ADDR, ENV_MASTER_PORT, ENV_NODE_RANK])
    logging.info(f"Distributed training: {env_vars}")

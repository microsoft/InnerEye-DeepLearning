#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import inspect
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import conda_merge
import ruamel.yaml
from health_azure.utils import (
    CONDA_CHANNELS, CONDA_DEPENDENCIES, CONDA_NAME, CONDA_PIP, CondaDependencies, PinnedOperator,
    _log_conda_dependencies_stats, _retrieve_unique_deps, is_conda_file_with_pip_include, is_pip_include_dependency
)

from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.Common.type_annotations import PathOrString
from InnerEye.ML.common import ModelExecutionMode

MAX_PATH_LENGTH = 260

# convert string to None if an empty string or whitespace is provided
empty_string_to_none = lambda x: None if (x is None or len(x.strip()) == 0) else x
string_to_path = lambda x: None if (x is None or len(x.strip()) == 0) else Path(x)

SUBJECT_METRICS_FILE_NAME = "metrics.csv"
EPOCH_METRICS_FILE_NAME = "epoch_metrics.csv"
METRICS_AGGREGATES_FILE = "metrics_aggregates.csv"
CROSSVAL_RESULTS_FOLDER = "CrossValResults"
BASELINE_COMPARISONS_FOLDER = "BaselineComparisons"
FULL_METRICS_DATAFRAME_FILE = "MetricsAcrossAllRuns.csv"
BEST_EPOCH_FOLDER_NAME = "best_validation_epoch"

OTHER_RUNS_SUBDIR_NAME = "OTHER_RUNS"
ENSEMBLE_SPLIT_NAME = "ENSEMBLE"

SCATTERPLOTS_SUBDIR_NAME = "scatterplots"
BASELINE_WILCOXON_RESULTS_FILE = "BaselineComparisonWilcoxonSignedRankTestResults.txt"


class ModelProcessing(Enum):
    """
    Enum used in model training and inference, used to decide where to put files and what logging messages to
    print. The meanings of the values are:

      * ``ENSEMBLE_CREATION``: we are creating and processing an ensemble model from within the child run with
        cross-validation index 0 of the HyperDrive run that created this model.
      * ``DEFAULT``: any other situation, *including* where the model is an ensemble model created by an earlier run
        (so the current run is standalone, not part of a HyperDrive run).

    There are four scenarios, only one of which uses ``ModelProcessing.ENSEMBLE_CREATION``:

        1. Training and inference on a single model in a single (non-HyperDrive) run.
        2. Training and inference on a single model that is part of an ensemble, in HyperDrive child run.
        3. Inference on an ensemble model taking place in a HyperDrive child run that trained one of the component
           models of the ensemble and whose cross validation index is 0.
        4. Inference on a single or ensemble model created in an another run specified by the value of run_recovery_id.

    The scenarios occur under the following conditions:

        * Scenario 1 happens when we train a model (``train=True``) with ``number_of_cross_validation_splits=0``. In
          this case, the value of ModelProcessing passed around is DEFAULT.
        * Scenario 2 happens when we train a model (``train=True``) with ``number_of_cross_validation_splits > 0``. In
          this case, the value of ModelProcessing passed around is DEFAULT in each of the child runs while training and
          running inference on its own single model. However, the child run whose cross validation index is 0 then goes
          on to carry out Scenario 3, and does more processing with ModelProcessing value ``ENSEMBLE_CREATION``, to
          create and register the ensemble model, run inference on it, and upload information about the ensemble model
          to the parent run.
        * Scenario 4 happens when we do an inference-only run (``train=False``), and specify an existing model with
          ``run_recovery_id`` (and necessarily ``number_of_cross_validation_splits=0``, even if the recovered run was a
          HyperDrive one). This model may be either a single one or an ensemble one; in both cases, a ModelProcessing
          value of ``DEFAULT`` is used.
    """
    DEFAULT = 'default'
    ENSEMBLE_CREATION = 'ensemble_creation'


def get_best_epoch_results_path(mode: ModelExecutionMode,
                                model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> Path:
    """
    For a given model execution mode, creates the relative results path
    in the form BEST_EPOCH_FOLDER_NAME/(Train, Test or Val)

    :param mode: model execution mode
    :param model_proc: whether this is for an ensemble or single model. If ensemble, we return a different path
        to avoid colliding with the results from the single model that may have been created earlier in the same run.
    """
    subpath = Path(BEST_EPOCH_FOLDER_NAME) / mode.value
    if model_proc == ModelProcessing.ENSEMBLE_CREATION:
        return Path(OTHER_RUNS_SUBDIR_NAME) / ENSEMBLE_SPLIT_NAME / subpath
    else:
        return subpath


def any_smaller_or_equal_than(items: Iterable[Any], scalar: float) -> bool:
    """
    Returns True if any of the elements of the list is smaller than the given scalar number.
    """
    return any(item < scalar for item in items)


def any_pairwise_larger(items1: Any, items2: Any) -> bool:
    """
    Returns True if any of the elements of items1 is larger than the corresponding element in items2.
    The two lists must have the same length.
    """
    if len(items1) != len(items2):
        raise ValueError(f"Arguments must have the same length. len(items1): {len(items1)}, len(items2): {len(items2)}")
    for i in range(len(items1)):
        if items1[i] > items2[i]:
            return True
    return False


def check_is_any_of(message: str, actual: Optional[str], valid: Iterable[Optional[str]]) -> None:
    """
    Raises an exception if 'actual' is not any of the given valid values.

    :param message: The prefix for the error message.
    :param actual: The actual value.
    :param valid: The set of valid strings that 'actual' is allowed to take on.
    """
    if actual not in valid:
        all_valid = ", ".join(["<None>" if v is None else v for v in valid])
        raise ValueError("{} must be one of [{}], but got: {}".format(message, all_valid, actual))


def get_items_from_string(string: str, separator: str = ',', remove_blanks: bool = True) -> List[str]:
    """
    Returns a list of items, separated by a known symbol, from a given string.
    """
    items = [item.strip() if remove_blanks else item for item in string.split(separator)]
    if remove_blanks:
        return list(filter(None, items))
    return items


logging_stdout_handler: Optional[logging.StreamHandler] = None
logging_to_file_handler: Optional[logging.StreamHandler] = None


def logging_to_stdout(log_level: Union[int, str] = logging.INFO) -> None:
    """
    Instructs the Python logging libraries to start writing logs to stdout up to the given logging level.
    Logging will use a timestamp as the prefix, using UTC.

    :param log_level: The logging level. All logging message with a level at or above this level will be written to
        stdout. log_level can be numeric, or one of the pre-defined logging strings (``loging.INFO``, ``logging.DEBUG``,
        etc.).
    """
    log_level = standardize_log_level(log_level)
    logger = logging.getLogger()
    # This function can be called multiple times, in particular in AzureML when we first run a training job and
    # then a couple of tests, which also often enable logging. This would then add multiple handlers, and repeated
    # logging lines.
    global logging_stdout_handler
    if not logging_stdout_handler:
        print("Setting up logging to stdout.")
        # At startup, logging has one handler set, that writes to stderr, with a log level of 0 (logging.NOTSET)
        if len(logger.handlers) == 1:
            logger.removeHandler(logger.handlers[0])
        logging_stdout_handler = logging.StreamHandler(stream=sys.stdout)
        _add_formatter(logging_stdout_handler)
        logger.addHandler(logging_stdout_handler)
    print(f"Setting logging level to {log_level} ({get_log_level_string(log_level)})")
    logging_stdout_handler.setLevel(log_level)
    logger.setLevel(log_level)


def standardize_log_level(log_level: Union[int, str]) -> int:
    """
    :param log_level: integer or string (any casing) version of a log level, e.g. 20 or "INFO".
    :return: integer version of the level; throws if the string does not name a level.
    """
    if isinstance(log_level, str):
        log_level = log_level.upper()
        check_is_any_of("log_level", log_level, logging._nameToLevel.keys())
        return logging._nameToLevel[log_level]
    return log_level


def get_log_level_string(log_level: int) -> str:
    """
    :param log_level: integer version of a log level, e.g. 20.
    :return: string version of the level; throws an error if the level is not registered.
    """
    valid_levels = list(logging._levelToName.keys())
    if log_level not in valid_levels:
        raise ValueError(f"Log level {log_level} is not valid. Possible values are: {valid_levels}")
    return logging._levelToName[log_level]


def logging_to_file(file_path: Path) -> None:
    """
    Instructs the Python logging libraries to start writing logs to the given file.
    Logging will use a timestamp as the prefix, using UTC. The logging level will be the same as defined for
    logging to stdout.

    :param file_path: The path and name of the file to write to.
    """
    # This function can be called multiple times, and should only add a handler during the first call.
    global logging_to_file_handler
    if not logging_to_file_handler:
        global logging_stdout_handler
        log_level = logging_stdout_handler.level if logging_stdout_handler else logging.INFO
        print(f"Setting up logging with level {log_level} to file {file_path}")
        file_path.parent.mkdir(exist_ok=True, parents=True)
        handler = logging.FileHandler(filename=str(file_path))
        _add_formatter(handler)
        handler.setLevel(log_level)
        logging.getLogger().addHandler(handler)
        logging_to_file_handler = handler


def disable_logging_to_file() -> None:
    """
    If logging to a file has been enabled previously via logging_to_file, this call will remove that logging handler.
    """
    global logging_to_file_handler
    if logging_to_file_handler:
        logging_to_file_handler.close()
        logging.getLogger().removeHandler(logging_to_file_handler)
        logging_to_file_handler = None


@contextmanager
def logging_only_to_file(file_path: Path, stdout_log_level: Union[int, str] = logging.ERROR) -> Generator:
    """
    Redirects logging to the specified file, undoing that on exit. If logging is currently going
    to stdout, messages at level stdout_log_level or higher (typically ERROR) are also sent to stdout.
    Usage: with logging_only_to_file(my_log_path): do_stuff()

    :param file_path: file to log to
    :param stdout_log_level: mininum level for messages to also go to stdout
    """
    stdout_log_level = standardize_log_level(stdout_log_level)
    logging_to_file(file_path)
    global logging_stdout_handler
    if logging_stdout_handler is not None:
        original_stdout_log_level = logging_stdout_handler.level
        logging_stdout_handler.level = stdout_log_level  # type: ignore
        yield
        logging_stdout_handler.level = original_stdout_log_level
    else:
        yield
    disable_logging_to_file()


def _add_formatter(handler: logging.StreamHandler) -> None:
    """
    Adds a logging formatter that includes the timestamp and the logging level.
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%SZ")
    # noinspection PyTypeHints
    formatter.converter = time.gmtime  # type: ignore
    handler.setFormatter(formatter)


@contextmanager
def logging_section(gerund: str) -> Generator:
    """
    Context manager to print "**** STARTING: ..." and "**** FINISHED: ..." lines around sections of the log,
    to help people locate particular sections. Usage:
    with logging_section("doing this and that"):
       do_this_and_that()

    :param gerund: string expressing what happens in this section of the log.
    """
    from time import time
    logging.info("")
    msg = f"**** STARTING: {gerund} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")
    start_time = time()
    yield
    elapsed = time() - start_time
    logging.info("")
    if elapsed >= 3600:
        time_expr = f"{elapsed / 3600:0.2f} hours"
    elif elapsed >= 60:
        time_expr = f"{elapsed / 60:0.2f} minutes"
    else:
        time_expr = f"{elapsed:0.2f} seconds"
    msg = f"**** FINISHED: {gerund} after {time_expr} "
    logging.info(msg + (100 - len(msg)) * "*")
    logging.info("")


def is_windows() -> bool:
    """
    Returns True if the host operating system is Windows.
    """
    return os.name == 'nt'


def is_linux() -> bool:
    """
    Returns True if the host operating system is a flavour of Linux.
    """
    return os.name == 'posix'


def check_properties_are_not_none(obj: Any, ignore: Optional[List[str]] = None) -> None:
    """
    Checks to make sure the provided object has no properties that have a None value assigned.
    """
    if ignore is not None:
        none_props = [k for k, v in vars(obj).items() if v is None and k not in ignore]
        if len(none_props) > 0:
            raise ValueError("Properties had None value: {}".format(none_props))


def initialize_instance_variables(func: Callable) -> Callable:
    """
    Automatically assigns the input parameters. Example usage::

        class process:
            @initialize_instance_variables
            def __init__(self, cmd, reachable=False, user='root'):
                pass
        p = process('halt', True)
        print(p.cmd, p.reachable, p.user)

    Outputs::

        ('halt', True, 'root')

    """
    names, varargs, keywords, defaults, _, _, _ = inspect.getfullargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):  # type: ignore
        # noinspection PyTypeChecker
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


def is_long_path(path: PathOrString) -> bool:
    """
    A long path is a path that has more than 260 characters
    """
    return len(str(path)) > MAX_PATH_LENGTH


def is_private_field_name(name: str) -> bool:
    """
    A private field is any Python class member that starts with an underscore eg: _hello
    """
    return name.startswith("_")


def is_gpu_tensor(data: Any) -> bool:
    import torch
    """
    Helper utility to check if the provided object is a GPU tensor
    """
    return data is not None and torch.is_tensor(data) and data.is_cuda


def print_exception(ex: Exception, message: str, logger_fn: Callable = logging.error) -> None:
    """
    Prints information about an exception, and the full traceback info.

    :param ex: The exception that was caught.
    :param message: An additional prefix that is printed before the exception itself.
    :param logger_fn: The logging function to use for logging this exception
    """
    logger_fn(f"{message} Exception: {ex}")
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)


def namespace_to_path(namespace: str, root: PathOrString = repository_root_directory()) -> Path:
    """
    Given a namespace (in form A.B.C) and an optional root directory R, create a path R/A/B/C

    :param namespace: Namespace to convert to path
    :param root: Path to prefix (default is project root)
    """""
    return Path(root, *namespace.split("."))


def path_to_namespace(path: Path, root: PathOrString = repository_root_directory()) -> str:
    """
    Given a path (in form R/A/B/C) and an optional root directory R, create a namespace A.B.C.
    If root is provided, then path must be a relative child to it.

    :param path: Path to convert to namespace
    :param root: Path prefix to remove from namespace (default is project root)
    :return: String representation to path of namespace
    """
    return ".".join([Path(x).stem for x in path.relative_to(root).parts])


def remove_file_or_directory(pth: Path) -> None:
    """
    Remove a directory and its contents, or a file.
    """
    if pth.is_dir():
        for child in pth.glob('*'):
            if child.is_file():
                child.unlink()
            else:
                remove_file_or_directory(child)
        pth.rmdir()
    elif pth.exists():
        pth.unlink()


@contextmanager
def change_working_directory(path_or_str: PathOrString) -> Generator:
    """
    Context manager for changing the current working directory
    """
    new_path = Path(path_or_str).expanduser()
    old_path = Path.cwd()
    os.chdir(new_path)
    yield
    os.chdir(old_path)


def merge_conda_files(
    conda_files: List[Path],
    result_file: Path,
    pip_files: Optional[List[Path]] = None,
) -> None:
    """
    Merges the given Conda environment files using the conda_merge package, optionally adds any
    dependencies from pip requirements files, and writes the merged file to disk.

    :param conda_files: The Conda environment files to read.
    :param result_file: The location where the merge results should be written.
    :param pip_files: An optional list of one or more pip requirements files including extra dependencies.
    """
    env_definitions: List[Any] = []
    for file in conda_files:
        _, pip_without_include = is_conda_file_with_pip_include(file)
        env_definitions.append(pip_without_include)
    unified_definition = {}

    extra_pip_deps = []
    for pip_file in pip_files or []:
        additional_pip_deps = [d for d in pip_file.read_text().split("\n") if d and not is_pip_include_dependency(d)]
        extra_pip_deps.extend(additional_pip_deps)

    name = conda_merge.merge_names(env.get(CONDA_NAME) for env in env_definitions)
    if name:
        unified_definition[CONDA_NAME] = name

    try:
        channels = conda_merge.merge_channels(env.get(CONDA_CHANNELS) for env in env_definitions)
    except conda_merge.MergeError:
        logging.error("Failed to merge channel priorities.")
        raise
    if channels:
        unified_definition[CONDA_CHANNELS] = channels

    try:
        deps_to_merge = [env.get(CONDA_DEPENDENCIES) for env in env_definitions]
        if len(extra_pip_deps) > 0:
            deps_to_merge.append([{CONDA_PIP: extra_pip_deps}])
        deps = conda_merge.merge_dependencies(deps_to_merge)

        # Get conda dependencies and pip dependencies from specification
        pip_deps_entries = [d for d in deps if isinstance(d, dict) and CONDA_PIP in d]  # type: ignore
        if len(pip_deps_entries) == 0:
            raise ValueError("Didn't find a dictionary with the key 'pip' in the list of dependencies")
        pip_deps_entry: Dict[str, List[str]] = pip_deps_entries[0]
        pip_deps = pip_deps_entry[CONDA_PIP]
        # temporarily remove pip dependencies from deps to be added back after deduplicaton
        deps.remove(pip_deps_entry)

        # remove all non-pip duplicates from the list of dependencies
        unique_deps = _retrieve_unique_deps(deps, PinnedOperator.CONDA)

        unique_pip_deps = sorted(_retrieve_unique_deps(pip_deps, PinnedOperator.PIP))

        # finally add back the deduplicated list of dependencies
        unique_deps.append({CONDA_PIP: unique_pip_deps})  # type: ignore

    except conda_merge.MergeError:
        logging.error("Failed to merge dependencies.")
        raise
    if unique_deps:
        unified_definition[CONDA_DEPENDENCIES] = unique_deps
    else:
        raise ValueError("No dependencies found in any of the conda files.")

    with result_file.open("w") as f:
        ruamel.yaml.dump(unified_definition, f, indent=2, default_flow_style=False)
    _log_conda_dependencies_stats(CondaDependencies(result_file), "Merged Conda environment")

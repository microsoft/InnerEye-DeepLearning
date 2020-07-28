#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import inspect
import logging
import os
import shutil
import sys
import time
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from InnerEye.Common.fixed_paths import repository_root_directory
from InnerEye.Common.type_annotations import PathOrString
from InnerEye.ML.common import ModelExecutionMode

MAX_PATH_LENGTH = 260

# convert string to None if an empty string or whitespace is provided
empty_string_to_none = lambda x: None if (x is None or len(x.strip()) == 0) else x
string_to_path = lambda x: None if (x is None or len(x.strip()) == 0) else Path(x)

# File name pattern that will match anything returned by epoch_folder_name.
EPOCH_FOLDER_NAME_PATTERN = "epoch_[0-9][0-9][0-9]"

METRICS_FILE_NAME = "metrics.csv"
EPOCH_METRICS_FILE_NAME = "epoch_metrics.csv"
METRICS_AGGREGATES_FILE = "metrics_aggregates.csv"
CROSSVAL_RESULTS_FOLDER = "CrossValResults"
FULL_METRICS_DATAFRAME_FILE = "MetricsAcrossAllRuns.csv"


class DataframeLogger:
    """
    Single DataFrame logger for logging to CSV file
    """

    def __init__(self, csv_path: Path):
        self.records: List[Dict[str, Any]] = []
        self.csv_path = csv_path

    def add_record(self, record: Dict[str, Any]) -> None:
        self.records.append(record)

    def flush(self, log_info: bool = False) -> None:
        """
        Save the internal records to a csv file.
        :param log_info: Log INFO if log_info is True.
        """
        import pandas as pd
        if not self.csv_path.parent.is_dir():
            self.csv_path.parent.mkdir(parents=True)
        # Specifying columns such that the order in which columns appear matches the order in which
        # columns were added in the code.
        columns = self.records[0].keys() if len(self.records) > 0 else None
        df = pd.DataFrame.from_records(self.records, columns=columns)
        df.to_csv(self.csv_path, sep=',', mode='w', index=False)
        if log_info:
            logging.info(f"\n {df.to_string(index=False)}")


class MetricsDataframeLoggers:
    """
    Contains DataframeLogger instances for logging metrics to CSV during training and validation stages respectively
    """
    def __init__(self, outputs_folder: Path):
        self.outputs_folder = outputs_folder
        _train_root = self.outputs_folder / ModelExecutionMode.TRAIN.value
        _val_root = self.outputs_folder / ModelExecutionMode.VAL.value
        # training loggers
        self.train_subject_metrics = DataframeLogger(_train_root / METRICS_FILE_NAME)
        self.train_epoch_metrics = DataframeLogger(_train_root / EPOCH_METRICS_FILE_NAME)
        # validation loggers
        self.val_subject_metrics = DataframeLogger(_val_root / METRICS_FILE_NAME)
        self.val_epoch_metrics = DataframeLogger(_val_root / EPOCH_METRICS_FILE_NAME)
        self._all_metrics = [
            self.train_subject_metrics, self.train_epoch_metrics,
            self.val_subject_metrics, self.val_epoch_metrics
        ]

    def close_all(self) -> None:
        """
        Save all records for each logger to disk.
        """
        for x in self._all_metrics:
            x.flush()


def epoch_folder_name(epoch: int) -> str:
    """
    Returns a formatted name for the a given epoch number, padded with zeros to 3 digits.
    """
    return "epoch_{0:03d}".format(epoch)


def get_epoch_results_path(epoch: int, mode: ModelExecutionMode) -> str:
    """
    For a given model execution mode, and an epoch index, creates the relative results path
    in the form epoch_x/(Train, Test or Val)
    :param epoch: epoch number
    :param mode: model execution mode
    """
    return Path(epoch_folder_name(epoch)) / mode.value


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
        raise ValueError("Arguments must have the same length. len(items1): {}, len(items2): {}"
                         .format(len(items1), len(items2)))
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
    :return:
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
    stdout. log_level can be numeric, or one of the pre-defined logging strings (INFO, DEBUG, ...).
    """
    if isinstance(log_level, str):
        log_level = log_level.upper()
        check_is_any_of("log_level", log_level, logging._nameToLevel.keys())
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
    print(f"Setting logging level to {log_level}")
    logging_stdout_handler.setLevel(log_level)
    logger.setLevel(log_level)


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


def _add_formatter(handler: logging.StreamHandler) -> None:
    """
    Adds a logging formatter that includes the timestamp and the logging level.
    """
    formatter = logging.Formatter(fmt="%(asctime)s %(levelname)-8s %(message)s",
                                  datefmt="%Y-%m-%dT%H:%M:%SZ")
    # noinspection PyTypeHints
    formatter.converter = time.gmtime  # type: ignore
    handler.setFormatter(formatter)


def delete_and_remake_directory(folder: Union[str, Path]) -> None:
    """
    Delete the folder if it exists, and remakes it.
    """
    folder = str(folder)

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.makedirs(folder)


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
    Automatically assigns the input parameters.

    >>> class process:
    ...     @initialize_instance_variables
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> # noinspection PyUnresolvedReferences
    >>> p.cmd, p.reachable, p.user
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
    :return:
    """""
    return Path(root, *namespace.split("."))


def path_to_namespace(path: Path, root: PathOrString = repository_root_directory()) -> str:
    """
    Given a path (in form R/A/B/C) and an optional root directory R, create a namespace A.B.C.
    If root is provided, then path must be a relative child to it.
    :param path: Path to convert to namespace
    :param root: Path prefix to remove from namespace (default is project root)
    :return:
    """
    return ".".join([Path(x).stem for x in path.relative_to(root).parts])


def get_namespace_root(namespace: str) -> Optional[Path]:
    """
    Given a namespace (in the form foo.bar.baz), returns a Path for the member of sys.path
    that contains a directory of file for that namespace, or None if there is no such member.
    """
    for root in sys.path:
        path = namespace_to_path(namespace, root)
        path_py = path.parent / (path.name + '.py')
        if path.exists() or path_py.exists():
            return Path(root)
    return None

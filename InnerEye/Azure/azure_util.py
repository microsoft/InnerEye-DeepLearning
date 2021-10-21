#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import re
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from azureml.core import Experiment, Run, Workspace, get_run
from azureml.exceptions import UserErrorException

from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import SUBJECT_METRICS_FILE_NAME
from health_azure.utils import create_run_recovery_id

DEFAULT_CROSS_VALIDATION_SPLIT_INDEX = -1
EXPERIMENT_RUN_SEPARATOR = ":"
EFFECTIVE_RANDOM_SEED_KEY_NAME = "effective_random_seed"
RUN_RECOVERY_ID_KEY_NAME = "run_recovery_id"
RUN_RECOVERY_FROM_ID_KEY_NAME = "recovered_from"
IS_ENSEMBLE_KEY_NAME = "is_ensemble"
MODEL_ID_KEY_NAME = "model_id"
# The name of the key used to store the cross validation index of the dataset for the run
CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY = "cross_validation_split_index"
PARENT_RUN_ID_KEY_NAME = "parent_run_id"

# This is the folder structure that AzureML generates to store all results for an experiment run.
# azureml is the name of the container
AZUREML_RUN_FOLDER_PREFIX = "dcid."
AZUREML_RUN_FOLDER = "azureml/ExperimentRun/" + AZUREML_RUN_FOLDER_PREFIX

# Global variables for the Run context, to avoid repeated HTTP calls to get it.
RUN_CONTEXT = Run.get_context()
# The Run context of the Hyperdrive parent run. This must be cached to avoid issues with the AzureML SDK,
# which creates worker pools for each call to .parent.
PARENT_RUN_CONTEXT = getattr(RUN_CONTEXT, "parent", None)

INNEREYE_SDK_NAME = "innereye"
INNEREYE_SDK_VERSION = "1.0"


def split_recovery_id(id: str) -> Tuple[str, str]:
    """
    Splits a run ID into the experiment name and the actual run.
    The argument can be in the format 'experiment_name:run_id',
    or just a run ID like user_branch_abcde12_123. In the latter case, everything before the last
    two alphanumeric parts is assumed to be the experiment name.
    :param id:
    :return: experiment name and run name
    """
    components = id.strip().split(EXPERIMENT_RUN_SEPARATOR)
    if len(components) > 2:
        raise ValueError("recovery_id must be in the format: 'experiment_name:run_id', but got: {}".format(id))
    elif len(components) == 2:
        return components[0], components[1]
    else:
        recovery_id_regex = r"^(\w+)_\d+_[0-9a-f]+$|^(\w+)_\d+$"
        match = re.match(recovery_id_regex, id)
        if not match:
            raise ValueError("The recovery ID was not in the expected format: {}".format(id))
        return (match.group(1) or match.group(2)), id


def fetch_run(workspace: Workspace, run_recovery_id: str) -> Run:
    """
    Finds an existing run in an experiment, based on a recovery ID that contains the experiment ID
    and the actual RunId. The run can be specified either in the experiment_name:run_id format,
    or just the run_id.
    :param workspace: the configured AzureML workspace to search for the experiment.
    :param run_recovery_id: The Run to find. Either in the full recovery ID format, experiment_name:run_id
    or just the run_id
    :return: The AzureML run.
    """
    experiment, run = split_recovery_id(run_recovery_id)
    try:
        experiment_to_recover = Experiment(workspace, experiment)
    except Exception as ex:
        raise Exception(f"Unable to retrieve run {run} in experiment {experiment}: {str(ex)}")
    run_to_recover = fetch_run_for_experiment(experiment_to_recover, run)
    logging.info("Fetched run #{} {} from experiment {}.".format(run, run_to_recover.number, experiment))
    return run_to_recover


def fetch_run_for_experiment(experiment_to_recover: Experiment, run_id: str) -> Run:
    """
    :param experiment_to_recover: an experiment
    :param run_id: a string representing the Run ID of one of the runs of the experiment
    :return: the run matching run_id_or_number; raises an exception if not found
    """
    try:
        return get_run(experiment=experiment_to_recover, run_id=run_id, rehydrate=True)
    except Exception:
        available_runs = experiment_to_recover.get_runs()
        available_ids = ", ".join([run.id for run in available_runs])
        raise (Exception(
            "Run {} not found for experiment: {}. Available runs are: {}".format(
                run_id, experiment_to_recover.name, available_ids)))


def fetch_runs(experiment: Experiment, filters: List[str]) -> List[Run]:
    """
    Fetch the runs in an experiment.
    :param experiment: the experiment to fetch runs from
    :param filters: a list of run status to include. Must be subset of [Running, Completed, Failed, Canceled].
    :return: the list of runs in the experiment
    """
    exp_runs = list(experiment.get_runs())

    if len(filters) != 0:
        if set.issubset(set(filters), ["Running", "Completed", "Failed", "Canceled"]):
            runs = [run for run in exp_runs if run.status in filters]
            exp_runs = runs

    return exp_runs


def fetch_child_runs(run: Run, status: Optional[str] = None,
                     expected_number_cross_validation_splits: int = 0) -> List[Run]:
    """
    Fetch child runs for the provided runs that have the provided AML status (or fetch all by default)
    and have a run_recovery_id tag value set (this is to ignore superfluous AML infrastructure platform runs).
    :param run: parent run to fetch child run from
    :param status: if provided, returns only child runs with this status
    :param expected_number_cross_validation_splits: when recovering child runs from AML hyperdrive
    sometimes the get_children function fails to retrieve all children. If the number of child runs
    retrieved by AML is lower than the expected number of splits, we try to retrieve them manually.
    """
    if is_ensemble_run(run):
        run_recovery_id = run.get_tags().get(RUN_RECOVERY_FROM_ID_KEY_NAME, None)
        if run_recovery_id:
            run = fetch_run(run.experiment.workspace, run_recovery_id)
        elif PARENT_RUN_CONTEXT:
            run = PARENT_RUN_CONTEXT
    children_runs = list(run.get_children(tags=RUN_RECOVERY_ID_KEY_NAME))
    if 0 < expected_number_cross_validation_splits != len(children_runs):
        if 0 < expected_number_cross_validation_splits != len(children_runs):
            logging.warning(
                f"The expected number of child runs was {expected_number_cross_validation_splits}."
                f"Fetched only: {len(children_runs)} runs. Now trying to fetch them manually.")
            run_ids_to_evaluate = [f"{create_run_recovery_id(run)}_{i}"
                                   for i in range(expected_number_cross_validation_splits)]
            children_runs = [fetch_run(run.experiment.workspace, id) for id in run_ids_to_evaluate]
    if status is not None:
        children_runs = [child_run for child_run in children_runs if child_run.get_status() == status]
    return children_runs


def is_ensemble_run(run: Run) -> bool:
    """Checks if the run was an ensemble of multiple models"""
    return run.get_tags().get(IS_ENSEMBLE_KEY_NAME) == 'True'


def to_azure_friendly_string(x: Optional[str]) -> Optional[str]:
    """
    Given a string, ensure it can be used in Azure by replacing everything apart from a-zA-Z0-9_ with _,
    and replace multiple _ with a single _.
    """
    if x is None:
        return x
    else:
        return re.sub('_+', '_', re.sub(r'\W+', '_', x))


def to_azure_friendly_container_path(path: Path) -> str:
    """
    Converts a path an Azure friendly container path by replacing "\\", "//" with "/" so it can be in the form: a/b/c.
    :param path: Original path
    :return: Converted path
    """
    return str(path).replace("\\", "/").replace("//", "/").strip("/")


def is_offline_run_context(run_context: Run) -> bool:
    """
    Tells if a run_context is offline by checking if it has an experiment associated with it.
    :param run_context: Context of the run to check
    :return:
    """
    return not hasattr(run_context, 'experiment')


def get_run_context_or_default(run: Optional[Run] = None) -> Run:
    """
    Returns the context of the run, if run is not None. If run is None, returns the context of the current run.
    :param run: Run to retrieve context for. If None, retrieve ocntext of current run.
    :return: Run context
    """
    return run if run else Run.get_context()


def get_cross_validation_split_index(run: Run) -> int:
    """
    Gets the cross validation index from the run's tags or returns the default
    :param run: Run context from which to get index
    :return: The cross validation split index
    """
    if is_offline_run_context(run):
        return DEFAULT_CROSS_VALIDATION_SPLIT_INDEX
    else:
        return int(run.get_tags().get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, DEFAULT_CROSS_VALIDATION_SPLIT_INDEX))


def is_cross_validation_child_run(run: Run) -> bool:
    """
    Checks the provided run's tags to determine if it is a cross validation child run
    (which is the case if the split index >=0)
    :param run: Run to check.
    :return: True if cross validation run. False otherwise.
    """
    return get_cross_validation_split_index(run) > DEFAULT_CROSS_VALIDATION_SPLIT_INDEX


def strip_prefix(string: str, prefix: str) -> str:
    """
    Returns the string without the prefix if it has the prefix, otherwise the string unchanged.
    :param string: Input string.
    :param prefix: Prefix to remove from input string.
    :return: Input string with prefix removed.
    """
    if string.startswith(prefix):
        return string[len(prefix):]
    return string


def get_all_environment_files(project_root: Path) -> List[Path]:
    """
    Returns a list of all Conda environment files that should be used. This is firstly the InnerEye conda file,
    and possibly a second environment.yml file that lives at the project root folder.
    :param project_root: The root folder of the code that starts the present training run.
    :return: A list with 1 or 2 entries that are conda environment files.
    """
    innereye_yaml = fixed_paths.get_environment_yaml_file()
    project_yaml = project_root / fixed_paths.ENVIRONMENT_YAML_FILE_NAME
    files = [innereye_yaml]
    if innereye_yaml != project_yaml:
        files.append(project_yaml)
    return files


def tag_values_all_distinct(runs: List[Run], tag: str) -> bool:
    """
    Returns True iff the runs all have the specified tag and all the values are different.
    """
    seen = set()
    for run in runs:
        value = run.get_tags().get(tag, None)
        if value is None or value in seen:
            return False
        seen.add(value)
    return True


def is_parent_run(run: Run) -> bool:
    return PARENT_RUN_CONTEXT and run.id == PARENT_RUN_CONTEXT.id


def download_run_output_file(blob_path: Path,
                             destination: Path,
                             run: Run) -> Path:
    """
    Downloads a single file from the run's default output directory: DEFAULT_AML_UPLOAD_DIR ("outputs").
    For example, if blobs_path = "foo/bar.csv", then the run result file "outputs/foo/bar.csv" will be downloaded
    to <destination>/bar.csv (the directory will be stripped off).
    :param blob_path: The name of the file to download.
    :param run: The AzureML run to download the files from
    :param destination: Local path to save the downloaded blob to.
    :return: Destination path to the downloaded file(s)
    """
    blobs_prefix = str((fixed_paths.DEFAULT_AML_UPLOAD_DIR / blob_path).as_posix())
    destination = destination / blob_path.name
    logging.info(f"Downloading single file from run {run.id}: {blobs_prefix} -> {str(destination)}")
    try:
        run.download_file(blobs_prefix, str(destination), _validate_checksum=True)
    except Exception as ex:
        raise ValueError(f"Unable to download file '{blobs_prefix}' from run {run.id}") from ex
    return destination


def download_run_outputs_by_prefix(blobs_prefix: Path,
                                   destination: Path,
                                   run: Run) -> None:
    """
    Download all the blobs from the run's default output directory: DEFAULT_AML_UPLOAD_DIR ("outputs") that
    have a given prefix (folder structure). When saving, the prefix string will be stripped off. For example,
    if blobs_prefix = "foo", and the run has a file "outputs/foo/bar.csv", it will be downloaded to destination/bar.csv.
    If there is in addition a file "foo.txt", that file will be skipped.
    :param blobs_prefix: The prefix for all files in "outputs" that should be downloaded.
    :param run: The AzureML run to download the files from.
    :param destination: Local path to save the downloaded blobs to.
    """
    prefix_str = str((fixed_paths.DEFAULT_AML_UPLOAD_DIR / blobs_prefix).as_posix())
    logging.info(f"Downloading multiple files from run {run.id}: {prefix_str} -> {str(destination)}")
    # There is a download_files function, but that can time out when downloading several large checkpoints file
    # (120sec timeout for all files).
    for file in run.get_file_names():
        if file.startswith(prefix_str):
            target_path = file[len(prefix_str):]
            if target_path.startswith("/"):
                target_path = target_path[1:]
                logging.info(f"Downloading {file}")
                run.download_file(file, str(destination / target_path), _validate_checksum=True)
            else:
                logging.warning(f"Skipping file {file}, because the desired prefix {prefix_str} is not aligned with "
                                f"the folder structure")


def is_running_on_azure_agent() -> bool:
    """
    Returns True if the code appears to be running on an Azure build agent, and False otherwise.
    """
    # Guess by looking at the AGENT_OS variable, that all Azure hosted agents define.
    return bool(os.environ.get("AGENT_OS", None))


def get_comparison_baseline_paths(outputs_folder: Path,
                                  blob_path: Path, run: Run,
                                  dataset_csv_file_name: str) -> \
        Tuple[Optional[Path], Optional[Path]]:
    run_rec_id = run.id
    # We usually find dataset.csv in the same directory as metrics.csv, but we sometimes
    # have to look higher up.
    comparison_dataset_path: Optional[Path] = None
    comparison_metrics_path: Optional[Path] = None
    destination_folder = outputs_folder / run_rec_id / blob_path
    # Look for dataset.csv inside epoch_NNN/Test, epoch_NNN/ and at top level
    for blob_path_parent in step_up_directories(blob_path):
        try:
            comparison_dataset_path = download_run_output_file(
                blob_path_parent / dataset_csv_file_name, destination_folder, run)
            break
        except (ValueError, UserErrorException):
            logging.warning(f"cannot find {dataset_csv_file_name} at {blob_path_parent} in {run_rec_id}")
        except NotADirectoryError:
            logging.warning(f"{blob_path_parent} is not a directory")
            break
        if comparison_dataset_path is None:
            logging.warning(f"cannot find {dataset_csv_file_name} at or above {blob_path} in {run_rec_id}")
    # Look for epoch_NNN/Test/metrics.csv
    try:
        comparison_metrics_path = download_run_output_file(
            blob_path / SUBJECT_METRICS_FILE_NAME, destination_folder, run)
    except (ValueError, UserErrorException):
        logging.warning(f"cannot find {SUBJECT_METRICS_FILE_NAME} at {blob_path} in {run_rec_id}")
    return (comparison_dataset_path, comparison_metrics_path)


def step_up_directories(path: Path) -> Generator[Path, None, None]:
    """
    Generates the provided directory and all its parents. Needed because dataset.csv
    files are sometimes not where we expect them to be, but higher up.
    """
    while True:
        yield path
        parent = path.parent
        if parent == path:
            break
        path = parent

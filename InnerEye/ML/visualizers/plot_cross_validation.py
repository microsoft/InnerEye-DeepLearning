#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
A script to download HyperDrive results from AzureML for cross validation (or single) runs to:
1) Create box-plot comparisons for each structure and split.
2) Create a list of outliers based on their distance from the mean in terms of standard deviation.
3) Create a query to identify the outliers in the production portal.

Statistical tests (Wilcoxon signed-rank and Mann-Whitney) are run on the downloaded data where appropriate,
to determine where one run is significantly better than another. These can be extended to cover more runs
than just the one whose results are plotted, by supplying values for --comparison_run_recovery_ids and
--comparison_epochs.
"""
import logging
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import param
import seaborn
from azureml.core import Run
from matplotlib import pyplot

import InnerEye.Common.Statistics.mann_whitney_test as mann_whitney
from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, download_outputs_from_run, \
    fetch_child_runs, \
    fetch_run, is_offline_run_context, is_parent_run
from InnerEye.Common import common_util, fixed_paths
from InnerEye.Common.Statistics.wilcoxon_signed_rank_test import WilcoxonTestConfig, wilcoxon_signed_rank_test
from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, DataframeLogger, ENSEMBLE_SPLIT_NAME, \
    FULL_METRICS_DATAFRAME_FILE, \
    METRICS_AGGREGATES_FILE, OTHER_RUNS_SUBDIR_NAME, logging_section, logging_to_stdout
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.Common.metrics_dict import INTERNAL_TO_LOGGING_COLUMN_NAMES, ScalarMetricsDict
from InnerEye.Common.type_annotations import PathOrString
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.deep_learning_config import DeepLearningConfig, ModelCategory
from InnerEye.ML.model_testing import METRICS_FILE_NAME, get_epoch_results_path
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SERIES_HEADER, CSV_SUBJECT_HEADER, OutlierType, \
    extract_outliers
from InnerEye.ML.utils.metrics_constants import LoggingColumns, MetricsFileColumns
from InnerEye.ML.visualizers.metrics_scatterplot import write_to_scatterplot_directory

RUN_DICTIONARY_NAME = "RunDictionary.txt"

MAX_STRUCTURES_PER_PLOT = 7
DRIVER_LOG_BASENAME = "70_driver_log.txt"
RUN_RECOVERY_ID_KEY = 'run_recovery_id'
# noinspection SQL
PORTAL_QUERY_TEMPLATE = "SELECT * FROM ROOT as r WHERE true AND ({}) AND ({})"
WILCOXON_RESULTS_FILE = "CrossValidationWilcoxonSignedRankTestResults.txt"
MANN_WHITNEY_RESULTS_FILE = "CrossValidationMannWhitneyTestResults.txt"
METRICS_BY_MODE_AND_STRUCTURE_FILE = "ResultsByModeAndStructure.csv"
METRICS_BY_MODE_FILE = "ResultsByMode.csv"

COL_SPLIT = "split"
COL_MODE = "mode"
FLOAT_FORMAT = "%.3f"

DEFAULT_PD_DISPLAY_CONTEXT = pd.option_context('display.max_colwidth', -1,
                                               'display.max_columns', None,
                                               'display.max_rows', None,
                                               'display.expand_frame_repr', False)

metric_types: Dict[str, Dict[str, Any]] = {
    MetricsFileColumns.Dice.value: {
        "title": 'Dice Score (%)',
        "ylim": [0, 100.5],
        "yticks": range(0, 105, 5),
        "outlier_type": OutlierType.LOW
    },
    MetricsFileColumns.HausdorffDistanceMM.value: {
        "title": 'Hausdorff Distance (mm)',
        "ylim": [0, 1000.0],
        "outlier_type": OutlierType.HIGH
    }
}


class PlotCrossValidationConfig(GenericConfig):
    """
    Configurations required to download results from the children of a HyperDrive runs.
    """
    model_category: ModelCategory = param.ClassSelector(class_=ModelCategory,
                                                        default=ModelCategory.Segmentation,
                                                        doc="The high-level model category described by this config.")
    run_recovery_id: Optional[str] = param.String(default=None, allow_None=True,
                                                  doc="The run recovery id of the run to collect results from."
                                                      "If the run is an ensemble, then the results for each"
                                                      "model will also be collected. This can be None in unit testing.")
    epoch: Optional[int] = param.Integer(default=None, allow_None=True, bounds=(1, None),
                                         doc="The epoch for which to fetch results")
    comparison_run_recovery_ids: List[str] = param.List(default=None, class_=str,
                                                        doc="The run recovery ids of any additional runs to include in "
                                                            "statistical comparisons")
    comparison_labels: List[str] = param.List(default=None, class_=str,
                                              doc="Short labels to use in plots for comparison runs")
    comparison_epochs: List[int] = param.List(default=None, class_=int,
                                              doc="The epochs of any additional runs to include in "
                                                  "statistical comparisons")
    compare_all_against_all: bool = param.Boolean(default=False,
                                                  doc="If set, include comparisons of comparison runs against "
                                                      "each other")
    outputs_directory: Path = param.ClassSelector(class_=Path, default=Path("."),
                                                  doc="The path to store results and get results "
                                                      "of plotting results for the current run")
    outlier_range: float = param.Number(3.0, doc="Number of standard deviations away from the mean to "
                                                 "use for outlier range")
    wilcoxon_test_p_value: float = param.Number(0.05, doc="Threshold for statistical tests")
    evaluation_set_name: str = param.String("test", doc="Set to run statistical tests on")
    ignore_subjects: List[int] = param.List(None, class_=int, bounds=(1, None), allow_None=True, instantiate=False,
                                            doc="List of the subject ids to ignore from the results")
    is_zero_index: bool = param.Boolean(True, doc="If True, start cross validation split indices from 0 otherwise 1")
    settings_yaml_file: Path = param.ClassSelector(class_=Path, default=fixed_paths.SETTINGS_YAML_FILE,
                                                   doc="Path to settings.yml file containing the Azure configuration "
                                                       "for the workspace")
    project_root: Path = param.ClassSelector(class_=Path, default=fixed_paths.repository_root_directory(),
                                             doc="The root folder of the repository that starts the run. Used to "
                                                 "read a private settings file.")
    _azure_config: Optional[AzureConfig] = \
        param.ClassSelector(class_=AzureConfig, allow_None=True,
                            doc="Azure-related options created from YAML file.")
    local_run_results: Optional[str] = \
        param.String(default=None, allow_None=True,
                     doc="Run results on local disk that can be used instead of accessing Azure in unit testing.")
    local_run_result_split_suffix: Optional[str] = \
        param.String(default=None, allow_None=True,
                     doc="Sub-folder of run results on local disk for each split, when doing unit testing.")
    number_of_cross_validation_splits: int = param.Integer(default=0,
                                                           doc="The expected number of splits in the"
                                                               "cross-validation run")
    create_plots: bool = param.Boolean(default=True, doc="Whether to create plots; if False, just find outliers "
                                                         "and do statistical tests")

    def __init__(self, **params: Any):
        # Mapping from run IDs to short names used in graphs
        self.short_names: Dict[str, str] = {}
        self.run_id_labels: Dict[str, str] = {}
        super().__init__(**params)

    def validate(self) -> None:
        if not self.run_recovery_id:
            raise ValueError("--run_recovery_id is a mandatory parameter.")
        if self.model_category == ModelCategory.Segmentation and self.epoch is None:
            raise ValueError("When working on segmentation models, --epoch is a mandatory parameter.")
        if self.comparison_run_recovery_ids is not None:
            # Extend comparison_epochs to be the same length as comparison_run_recovery_ids, using
            # the value of --epoch as the default if no value at all is given
            if self.comparison_epochs is None:
                self.comparison_epochs = [self.epoch]
            n_needed = len(self.comparison_run_recovery_ids) - len(self.comparison_epochs)
            if n_needed > 0:
                self.comparison_epochs.extend([self.comparison_epochs[-1]] * n_needed)
        else:
            self.comparison_run_recovery_ids = []
        if self.comparison_labels is None:
            self.comparison_labels = []
        self.run_id_labels[self.run_recovery_id] = "FOCUS"
        for run_id, label in zip(self.comparison_run_recovery_ids, self.comparison_labels):
            self.run_id_labels[run_id] = label

    def get_short_name(self, run_or_id: Union[Run, str]) -> str:
        if isinstance(run_or_id, Run):
            run_id = run_or_id.id
            if run_id not in self.short_names:
                if run_id in self.run_id_labels:
                    extra = f" ({self.run_id_labels[run_id]})"
                else:
                    extra = ""
                self.short_names[run_id] = f"{run_or_id.experiment.name}:{run_or_id.number}{extra}"
        else:
            run_id = run_or_id.split(":")[-1]
            if run_id not in self.short_names:
                self.short_names[run_or_id] = run_id
        return self.short_names[run_id]

    def execution_modes_to_download(self) -> List[ModelExecutionMode]:
        if self.model_category.is_scalar:
            return [ModelExecutionMode.TRAIN, ModelExecutionMode.VAL, ModelExecutionMode.TEST]
        else:
            return [ModelExecutionMode.VAL, ModelExecutionMode.TEST]

    @property
    def azure_config(self) -> AzureConfig:
        """
        Gets the AzureConfig instance that the script uses.
        :return:
        """
        if self._azure_config is None:
            self._azure_config = AzureConfig.from_yaml(self.settings_yaml_file, project_root=self.project_root)
        return self._azure_config

    def download_or_get_local_file(self,
                                   run: Optional[Run],
                                   blob_to_download: PathOrString,
                                   destination: Path,
                                   local_src_subdir: Optional[Path] = None) -> Optional[Path]:
        """
        Downloads a file from the results folder of an AzureML run, or copies it from a local results folder.
        Returns the path to the downloaded file if it exists, or None if the file was not found.
        If the blobs_path contains folders, the same folder structure will be created inside the destination folder.
        For example, downloading "foo.txt" to "/c/temp" will create "/c/temp/foo.txt". Downloading "foo/bar.txt"
        to "/c/temp" will create "/c/temp/foo/bar.txt"
        :param blob_to_download: path of data to download within the run
        :param destination: directory to write to
        :param run: The AzureML run to download from.
        :param local_src_subdir: if not None, then if we copy from a local results folder, that folder is
        self.outputs_directory/local_src_subdir/blob_to_download instead of self.outputs_directory/blob_to_download
        :return: The path to the downloaded file, or None if the file was not found.
        """
        blob_path = Path(blob_to_download)
        blob_parent = blob_path.parent
        if blob_parent != Path("."):
            destination = destination / blob_parent
        downloaded_file = destination / blob_path.name
        # If we've already downloaded the data, leave it as it is
        if downloaded_file.exists():
            logging.info(f"Download of '{blob_path}' to '{downloaded_file}: not needed, already exists'")
            return downloaded_file
        logging.info(f"Download of '{blob_path}' to '{downloaded_file}': proceeding")
        # If the provided run is the current run, then there is nothing to download.
        # Just copy the provided path in the outputs directory to the destination.
        if not destination.exists():
            destination.mkdir(parents=True)
        if run is None or Run.get_context().id == run.id or is_parent_run(run) or is_offline_run_context(run):
            if run is None:
                assert self.local_run_results is not None, "Local run results must be set in unit testing"
                local_src = Path(self.local_run_results)
                if self.local_run_result_split_suffix:
                    local_src = local_src / self.local_run_result_split_suffix
            else:
                local_src = self.outputs_directory
            if local_src_subdir is not None:
                local_src = local_src / local_src_subdir
            local_src = local_src / blob_path
            if local_src.exists():
                logging.info(f"Copying files from {local_src} to {destination}")
                return Path(shutil.copy(local_src, destination))
            return None
        else:
            try:
                return download_outputs_from_run(
                    blobs_path=blob_path,
                    destination=destination,
                    run=run,
                    is_file=True
                )
            except Exception as ex:
                logging.warning(f"File {blob_to_download} not found in output of run {run.id}: {ex}")
                return None


@dataclass(frozen=True)
class RunResultFiles:
    execution_mode: ModelExecutionMode
    metrics_file: Path
    dataset_csv_file: Optional[Path]
    run_recovery_id: Optional[str]
    split_index: str

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self, ignore=["dataset_csv_file", "run_recovery_id"])


@dataclass(frozen=True)
class OfflineCrossvalConfigAndFiles:
    """
    Stores a configuration for crossvalidation analysis, and all the required input files.
    """
    config: PlotCrossValidationConfig
    files: List[RunResultFiles]

    def __post_init__(self) -> None:
        common_util.check_properties_are_not_none(self)


@dataclass(frozen=True)
class EpochMetricValues:
    epoch: int
    metric_name: str
    metric_value: float


def get_split_id(tags: Dict[str, Any], is_zero_index: bool = True) -> str:
    """
    Extracts the split index from the tags. If it's negative, this isn't a cross-validation run;
    gets it from the run_recovery_id instead.
    :param tags: Tags associated with a run to get the split id for
    :param is_zero_index: If True, start cross validation split indices from 0 otherwise 1
    :return:
    """
    index = int(tags.get(CROSS_VALIDATION_SPLIT_INDEX_TAG_KEY, -1))
    if index < 0:
        return run_recovery_id_suffix(tags)
    else:
        return str(index if is_zero_index else index + 1)


def run_recovery_id_suffix(tags: Dict[str, Any]) -> str:
    """
    Returns the part of run_recovery_id after the colon if any.
    :param tags: the tags of a run
    """
    run_rec_id = tags[RUN_RECOVERY_ID_KEY]
    return run_rec_id.split(':')[-1]


def download_metrics_file(config: PlotCrossValidationConfig,
                          run: Run,
                          destination: Path,
                          epoch: Optional[int],
                          mode: ModelExecutionMode) -> Optional[Path]:
    """
    Downloads a metrics.csv file from an Azure run (or local results), and stores it in a local folder.
    The metrics.csv file will be written into a subfolder named after the model execution mode.
    :param config: The cross validation configuration.
    :param run: The AzureML run to download from.
    :param destination: The folder to download into.
    :param epoch: The epoch that plot_cross_validation is running for. This is mandatory for segmentation models,
    and ignored for classification models.
    :param mode: The dataset split to read from.
    :return: The path to the local file, or None if no metrics.csv file was found.
    """
    # setup the appropriate paths and readers for the metrics
    if config.model_category == ModelCategory.Segmentation:
        if epoch is None:
            raise ValueError("Epoch must be provided in segmentation runs")
        src = get_epoch_results_path(epoch, mode) / METRICS_FILE_NAME
    else:
        src = Path(mode.value) / METRICS_FILE_NAME

    # download (or copy from local disc) subject level metrics for the given epoch
    local_src_subdir = Path(OTHER_RUNS_SUBDIR_NAME) / ENSEMBLE_SPLIT_NAME if is_parent_run(run) else None
    return config.download_or_get_local_file(
        blob_to_download=src,
        destination=destination,
        run=run,
        local_src_subdir=local_src_subdir)


def download_crossval_result_files(config: PlotCrossValidationConfig,
                                   run_recovery_id: Optional[str] = None,
                                   epoch: Optional[int] = None,
                                   download_to_folder: Optional[Path] = None,
                                   splits_to_evaluate: Optional[List[str]] = None) -> Tuple[List[RunResultFiles], Path]:
    """
    Given an AzureML run, downloads all files that are necessary for doing an analysis of cross validation runs.
    It will download the metrics.csv file for each dataset split (,Test, Val) and all of the run's children.
    When running in segmentation mode, it also downloads the dataset.csv and adds the institutionId and seriesId
    information for each subject found in the metrics files.
    :param config: PlotCrossValidationConfig
    :param run_recovery_id: run recovery ID, if different from the one in config
    :param epoch: epoch, if different from the one in config
    :param download_to_folder: The root folder in which all downloaded files should be stored. Point to an existing
    folder with downloaded files for use in unit tests. If not provided, the files will be downloaded to a new folder
    inside the config.outputs_directory, with the name taken from the run ID.
    :param splits_to_evaluate: If supplied, use these values as the split indices to download. Use only for
    unit testing.
    :return: The dataframe with all of the downloaded results grouped by execution mode (Test or Val)
     and directory where the epoch results were downloaded to.
    """
    splits_to_evaluate = splits_to_evaluate or []
    if run_recovery_id is None:
        run_recovery_id = config.run_recovery_id
    if epoch is None:
        epoch = config.epoch
    if run_recovery_id:
        workspace = config.azure_config.get_workspace()
        parent = fetch_run(workspace, run_recovery_id)
        runs_to_evaluate = fetch_child_runs(
            run=parent, expected_number_cross_validation_splits=config.number_of_cross_validation_splits)
        logging.info("Adding parent run to the list of runs to evaluate.")
        runs_to_evaluate.append(parent)
        logging.info(f"Will evaluate results for runs: {[x.id for x in runs_to_evaluate]}")
    else:
        runs_to_evaluate = []
    # create the root path to store the outputs
    if not download_to_folder:
        download_to_folder = config.outputs_directory / CROSSVAL_RESULTS_FOLDER
        # Make the folder if it doesn't exist, but preserve any existing contents.
        download_to_folder.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    logging.info(f"Starting to download files for cross validation analysis to: {download_to_folder}")
    assert download_to_folder is not None
    result: List[RunResultFiles] = []
    loop_over: List[Tuple[Optional[Run], str, str, Optional[str]]]
    if splits_to_evaluate:
        loop_over = [(None, split, split, "") for split in splits_to_evaluate]
    else:
        loop_over = []
        for run in runs_to_evaluate:
            tags = run.get_tags()
            if is_parent_run(run):
                split_index = ENSEMBLE_SPLIT_NAME
            else:
                split_index = get_split_id(tags, config.is_zero_index)
            split_suffix = split_index
            # Value to put in the "Split" column in the result.
            run_recovery_id = tags[RUN_RECOVERY_ID_KEY]
            loop_over.append((run, split_index, split_suffix, run_recovery_id))

    for run, split_index, split_suffix, run_recovery_id in loop_over:
        if run is not None:
            config.get_short_name(run)
        config.local_run_result_split_suffix = split_suffix
        # When run is the parent run, we need to look on the local disc.
        # If (as expected) dataset.csv is not already present, we copy it from the top of the outputs directory.
        folder_for_run = download_to_folder / split_suffix
        dataset_file: Optional[Path]
        if is_parent_run(run):
            folder_for_run.mkdir(parents=True, exist_ok=True)
            dataset_file = folder_for_run / DATASET_CSV_FILE_NAME
            # Copy the run-0 dataset.csv, which should be the same, as the parent run won't have one.
            shutil.copy(str(config.outputs_directory / DATASET_CSV_FILE_NAME), str(dataset_file))
        else:
            dataset_file = config.download_or_get_local_file(run, DATASET_CSV_FILE_NAME, folder_for_run)
        if config.model_category == ModelCategory.Segmentation and not dataset_file:
            raise ValueError(f"Dataset file must be present for segmentation models, but is missing for run {run.id}")
        # Get metrics files.
        for mode in config.execution_modes_to_download():
            # download metrics.csv file for each split. metrics_file can be None if the file does not exist
            # (for example, if no output was written for execution mode Test)
            metrics_file = download_metrics_file(config, run, folder_for_run, epoch, mode)
            if metrics_file:
                result.append(RunResultFiles(execution_mode=mode,
                                             dataset_csv_file=dataset_file,
                                             metrics_file=metrics_file,
                                             run_recovery_id=run_recovery_id,
                                             split_index=split_index))
    elapsed = time.time() - start_time
    logging.info(f"Finished downloading files. Total time to download: {elapsed:0.2f}sec")
    return result, download_to_folder


def crossval_config_from_model_config(train_config: DeepLearningConfig) -> PlotCrossValidationConfig:
    """
    Creates a configuration for plotting cross validation results that populates some key fields from the given
    model training configuration.
    :param train_config:
    :return:
    """
    # Default to the last epoch for segmentation models. For classification models, the epoch does not need to be
    # specified because datafiles contain results for all epochs.
    epoch = train_config.num_epochs if train_config.is_segmentation_model else None

    return PlotCrossValidationConfig(
        run_recovery_id=None,
        model_category=train_config.model_category,
        epoch=epoch,
        should_validate=False,
        number_of_cross_validation_splits=train_config.get_total_number_of_cross_validation_runs())


def get_config_and_results_for_offline_runs(train_config: DeepLearningConfig) -> OfflineCrossvalConfigAndFiles:
    """
    Creates a configuration for crossvalidation analysis for the given model training configuration, and gets
    the input files required for crossvalidation analysis.
    :param train_config: The model configuration to work with.
    """
    plot_crossval_config = crossval_config_from_model_config(train_config)
    download_to_folder = train_config.outputs_folder / CROSSVAL_RESULTS_FOLDER
    plot_crossval_config.outputs_directory = download_to_folder
    plot_crossval_config.local_run_results = str(train_config.outputs_folder)

    splits = [str(i) for i in range(plot_crossval_config.number_of_cross_validation_splits)] \
        if train_config.perform_cross_validation else [""]
    result_files, _ = download_crossval_result_files(plot_crossval_config,
                                                     download_to_folder=download_to_folder,
                                                     splits_to_evaluate=splits)
    return OfflineCrossvalConfigAndFiles(config=plot_crossval_config,
                                         files=result_files)


def load_dataframes(result_files: List[RunResultFiles], config: PlotCrossValidationConfig) \
        -> Dict[ModelExecutionMode, Optional[pd.DataFrame]]:
    """
    From a list of run result files on the local disk, create a dictionary of aggregate metrics.
    The dictionary maps from execution mode to metrics for that execution mode. The metrics
    for all cross validation splits are concatenated.
    If sub fold cross validation is performed then the results for each sub fold are aggregated into their
    parent fold by averaging.
    The resulting dictionary can contains only the execution modes for which metrics are present.

    :param result_files: The list of files to read.
    :param config: The overall configuration for the cross validation analysis.
    :return: A dictionary that maps from model execution mode (TEST, VAL) to concatenated metrics.
    For segmentation models, the dataset dataframe is joined to the metrics, and the portal series
    ID and institution ID are added.
    """
    dataset_split_metrics: Dict[ModelExecutionMode, List[pd.DataFrame]] = \
        {mode: [] for mode in config.execution_modes_to_download()}

    for result in result_files:
        mode = result.execution_mode
        is_segmentation = config.model_category == ModelCategory.Segmentation
        df = load_metrics_df(result.metrics_file, is_segmentation, config.epoch, result.execution_mode)
        if df is not None:
            # for segmentation models dataset.csv also needs to be downloaded and joined with the metrics.csv
            if config.model_category == ModelCategory.Segmentation:
                assert result.dataset_csv_file is not None
                dataset_df: pd.DataFrame = pd.read_csv(str(result.dataset_csv_file))
                # Join the seriesId and institutionId columns in dataset.csv to metrics.csv
                df = convert_rows_for_comparisons(result.run_recovery_id, dataset_df, df, mode, config.ignore_subjects)
            dataset_split_metrics[mode].append(df)

    # concatenate the data frames per split, removing execution modes for which there is no data.
    combined_metrics = {k: pd.concat(v) for k, v in dataset_split_metrics.items() if v}

    if config.model_category.is_scalar:
        # if child folds are present then combine model outputs
        for k, v in combined_metrics.items():
            aggregation_column = LoggingColumns.ModelOutput.value
            group_by_columns = [x for x in v.columns if x != aggregation_column]
            combined_metrics[k] = v.groupby(group_by_columns, as_index=False)[aggregation_column].mean() \
                .sort_values(list(v.columns), ascending=True).reset_index(drop=True)
    return combined_metrics


def load_metrics_df(metrics_file: Path,
                    is_segmentation: bool,
                    epoch: Optional[int],
                    mode: ModelExecutionMode) -> pd.DataFrame:
    """
    Reads the given metrics.csv file, and returns the results for the provided epoch,
    and model execution mode. If no epoch is provided, the results for all epochs are returned
    (only for classification models, because the segmentation metrics files always only contain results for a single
    epoch)
    """
    # Read via Pandas. Convert all columns automatically apart from prediction target,
    # which should always remain string.
    csv = pd.read_csv(metrics_file,
                      converters={LoggingColumns.Hue.value: lambda x: x})
    if is_segmentation:
        # Ensure the model execution mode is set in the downloaded dataframe
        csv[LoggingColumns.DataSplit.value] = mode.value
    else:
        # Metrics files for classification store results for all modes and epochs, restrict to the desired ones.
        csv = csv[csv[LoggingColumns.DataSplit.value] == mode.value]
        if epoch:
            csv = csv[csv[LoggingColumns.Epoch.value] == epoch]
    return csv


def convert_rows_for_comparisons(split_column_value: Optional[str],
                                 dataset_df: pd.DataFrame,
                                 df: pd.DataFrame,
                                 mode: ModelExecutionMode = ModelExecutionMode.TEST,
                                 ignore_subjects: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Given a dataframe resulting from a metrics.csv form, return (a subset of) the same information
    in the format required for a multi-split metrics file as required for statistical testing.
    :param split_column_value: name of this split, for putting in the "Split" column of the result
    :param dataset_df: dataframe read from a dataset.csv file
    :param df: dataframe read from a metrics.csv file
    :param mode: which mode to keep
    :param ignore_subjects: List of subjects to drop
    :return: augmented subset of the rows in df, as described
    """
    pre_len = len(df)
    # We need the institution column to compare subjects across institutions, if it is not present with add a default
    # value
    if CSV_INSTITUTION_HEADER not in dataset_df.columns:
        dataset_df[CSV_INSTITUTION_HEADER] = ''
    df = pd.merge(df, dataset_df[[CSV_SUBJECT_HEADER, CSV_SERIES_HEADER, CSV_INSTITUTION_HEADER]],
                  left_on=MetricsFileColumns.Patient.value, right_on=CSV_SUBJECT_HEADER) \
        .drop_duplicates().drop([CSV_SUBJECT_HEADER], axis=1)
    post_len = len(df)
    if pre_len != post_len:
        raise ValueError(f"len(df) {pre_len} becomes {post_len} for split {split_column_value}")
    # Add new columns corresponding to the dataset split
    df[COL_SPLIT] = split_column_value
    df[MetricsFileColumns.Dice.value] *= 100
    df[COL_MODE] = mode.value
    if ignore_subjects:
        df = df.drop(df.loc[df[MetricsFileColumns.Patient.value].isin(ignore_subjects)].index)
    return df


def shorten_split_names(config: PlotCrossValidationConfig, metrics: pd.DataFrame) -> None:
    """
    Replaces values in metrics[COL_SPLIT] by shortened versions consisting of the first 3 and last
    3 characters, separated by "..", when that string is shorter.
    :param config: for finding short names
    :param metrics: data frame with a column named COL_SPLIT
    """
    metrics[COL_SPLIT] = metrics[COL_SPLIT].apply(config.get_short_name)


def split_by_structures(df: pd.DataFrame) -> List[pd.DataFrame]:
    names = sorted(df[MetricsFileColumns.Structure.value].unique())
    n_structures = len(names)
    n_plots = int((n_structures + MAX_STRUCTURES_PER_PLOT - 1) / MAX_STRUCTURES_PER_PLOT)
    if n_plots <= 1:
        return [df]
    n_per_plot = n_structures / n_plots
    df_list = []
    start = 0
    for idx in range(n_plots):
        end = int(0.5 + (idx + 1) * n_per_plot)
        names_to_use = names[start:end]
        df_list.append(df[df[MetricsFileColumns.Structure.value].isin(names_to_use)])
        start = end
    return df_list


def plot_metrics(config: PlotCrossValidationConfig,
                 dataset_split_metrics: Dict[ModelExecutionMode, pd.DataFrame], root: Path) -> None:
    """
    Given the dataframe for the downloaded metrics aggregate them
    into box plots corresponding to the results per split.
    :param config: PlotCrossValidationConfig
    :param dataset_split_metrics: Mapping between model execution mode and a dataframe containing all metrics for it
    :param root: Root directory to the results for Train/Test and Val datasets
    :return:
    """
    for mode, df in dataset_split_metrics.items():
        for metric_type, props in get_available_metrics(df).items():
            df_list = split_by_structures(df)
            for sub_df_index, sub_df in enumerate(df_list, 1):
                metrics: pd.DataFrame = pd.melt(sub_df, id_vars=[COL_SPLIT, MetricsFileColumns.Structure.value],
                                                value_vars=[metric_type])
                shorten_split_names(config, metrics)
                metrics = metrics.sort_values(by=[COL_SPLIT, MetricsFileColumns.Structure.value])
                # create plot for the dataframe
                fig, ax = pyplot.subplots(figsize=(15.7, 8.27))
                ax = seaborn.boxplot(x='split', y='value',
                                     hue='Structure',
                                     data=metrics,
                                     ax=ax,
                                     width=0.75,
                                     palette="Set3")
                if "ylim" in props:
                    ax.set_ylim([0, 100.5])
                if "yticks" in props:
                    ax.set_yticks(range(0, 105, 5))

                ax.set_ylabel(props["title"], fontsize=16)
                ax.set_xlabel('Cross-Validation Splits', fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=12)
                ax.tick_params(axis='both', which='minor', labelsize=10)
                ax.set_title("Segmentation Results on {} Dataset (Epoch {})".format(
                    mode.value, config.epoch), fontsize=16)
                ax.set_axisbelow(True)
                ax.grid()
                # Shrink current axis by 20%
                box = ax.get_position()
                ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                # Put a legend to the right of the current axis
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                          fancybox=True, shadow=True, ncol=7, prop={'size': 12})

                # save plot
                suffix = f"_{sub_df_index}" if len(df_list) > 1 else ""
                plot_dst = root / f"{metric_type}_{mode.value}_splits{suffix}.jpg"
                fig.savefig(plot_dst, bbox_inches='tight')
                logging.info("Saved box-plots to: {}".format(plot_dst))


def save_outliers(config: PlotCrossValidationConfig,
                  dataset_split_metrics: Dict[ModelExecutionMode, pd.DataFrame], root: Path) -> None:
    """
    Given the dataframe for the downloaded metrics identifies outliers (score < mean - 3sd) across the splits
    and saves them in a file outlier.csv in the provided root.
    :param config: PlotCrossValidationConfig
    :param dataset_split_metrics: Mapping between model execution mode and a dataframe containing all metrics for it
    :param root: Root directory to the results for Train/Test and Val datasets
    :return:
    """
    stats_columns = ['count', 'mean', 'min', 'max']
    for mode, df in dataset_split_metrics.items():
        outliers_std = str(root / "{}_outliers.txt".format(mode.value))
        with open(outliers_std, 'w') as f:
            # to make sure no columns or rows are truncated
            with DEFAULT_PD_DISPLAY_CONTEXT:
                for metric_type, metric_type_metadata in get_available_metrics(df).items():
                    outliers = extract_outliers(
                        df=df,
                        outlier_range=config.outlier_range,
                        outlier_type=metric_type_metadata["outlier_type"]
                    ).drop([COL_SPLIT], axis=1)

                    f.write(f"\n\n=== METRIC: {metric_type} ===\n\n")
                    if len(outliers) > 0:
                        outliers_summary = str(outliers.groupby(
                            [MetricsFileColumns.Patient.value, MetricsFileColumns.Structure.value,
                             CSV_SERIES_HEADER, CSV_INSTITUTION_HEADER])
                                               .describe()[metric_type][stats_columns]
                                               .sort_values(stats_columns, ascending=False))
                        f.write(outliers_summary)
                        f.write("\n\n")
                        f.write(create_portal_query_for_outliers(outliers))
                    else:
                        f.write("No outliers found")

        print("Saved outliers to: {}".format(outliers_std))


def create_portal_query_for_outliers(df: pd.DataFrame) -> str:
    """
    Create a portal query string as a conjunction of the disjunctions of the unique InstitutionId and seriesId values.
    """
    return PORTAL_QUERY_TEMPLATE.format(
        " OR ".join(map(lambda x: 'r.InstitutionId = "{}"'.format(x), df[CSV_INSTITUTION_HEADER].unique())),
        " OR ".join(map(lambda x: 'STARTSWITH(r.VersionedDicomImageSeries.Latest.Series.InstanceUID,"{}")'.format(x),
                        df[CSV_SERIES_HEADER].unique()))
    )


def create_results_breakdown(df: pd.DataFrame, root_folder: Path) -> Tuple[Path, Path]:
    """
    Creates a breakdown of Dice per execution mode (train/test/val) and structure name, and one of
    Dice per execution mode. The summaries are saved to files in the root_folder, via dataframe's
    describe function.
    :param df: A data frame that contains columns COL_DICE, COL_MODE and COL_STRUCTURE
    :param root_folder: The folder into which the result files should be written.
    :return: The paths to the two files.
    """
    df = df[[COL_MODE, MetricsFileColumns.Structure.value, MetricsFileColumns.Dice.value]]
    file1 = root_folder / METRICS_BY_MODE_AND_STRUCTURE_FILE
    file2 = root_folder / METRICS_BY_MODE_FILE

    def _describe(_df: pd.DataFrame, group_by: List[str], file: Path) -> None:
        grouped = _df.groupby(group_by).describe()[MetricsFileColumns.Dice.value]
        columns = ["mean", "50%", "min", "max"]
        grouped[columns].to_csv(file, float_format=FLOAT_FORMAT)

    _describe(df, [COL_MODE, MetricsFileColumns.Structure.value], file1)
    _describe(df, [COL_MODE], file2)

    return file1, file2


def may_write_lines_to_file(lines: List[str], path: Path) -> None:
    """
    Prints lines to file path if there are any lines; reports what it's doing.
    :param lines: list of lines to write (without final newlines)
    :param path: csv file path to write to
    """
    with path.open('w') as out:
        if len(lines) == 0:
            logging.warning(f"Writing explanatory message to {path}")
            out.write("There were not enough data points for any statistically meaningful comparisons.")
        else:
            logging.info(f"Writing {len(lines)} lines to {path}")
            out.write("\n".join(lines) + "\n")


def run_statistical_tests_on_file(root_folder: Path, full_csv_file: Path, options: PlotCrossValidationConfig,
                                  focus_splits: Optional[List[str]]) -> None:
    """
    Runs a Wilcoxon signed-rank test to distinguish PAIRED scores between splits (i.e. when same subjects occur),
    and a Mann-Whitney test on (necessarily UNPAIRED) scores on subjects from different institutions in the
    same split. Empty results are not printed; this can happen for Wilcoxon when different splits do not
    intersect (e.g. when processing validation sets on a cross-val run) and for Mann-Whitney when only one
    institution occurs, or at most one institution has five or more subjects.
    :param root_folder: folder to write to
    :param full_csv_file: MetricsAcrossAllRuns.csv file to read
    :param options: config options.
    """
    against = None if options.compare_all_against_all else focus_splits
    config = WilcoxonTestConfig(csv_file=str(full_csv_file), with_scatterplots=options.create_plots, against=against,
                                subset=options.evaluation_set_name, exclude='')
    wilcoxon_lines, plots = wilcoxon_signed_rank_test(config, name_shortener=options.get_short_name)
    write_to_scatterplot_directory(root_folder, plots)
    may_write_lines_to_file(wilcoxon_lines, root_folder / WILCOXON_RESULTS_FILE)
    mann_whitney_lines = mann_whitney.compare_scores_across_institutions(
        str(full_csv_file), mode_to_use=options.evaluation_set_name)
    may_write_lines_to_file(mann_whitney_lines, root_folder / MANN_WHITNEY_RESULTS_FILE)


def get_available_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in metric_types.items() if k in df.columns}


def check_result_file_counts(config_and_files: OfflineCrossvalConfigAndFiles) -> None:
    """
    Check that for every ModelExecutionMode appearing in config_and_files.files, the number of files of
    that mode is equal to the number of cross-validation splits. Throw a ValueError if not.
    """
    result_files_by_mode = defaultdict(list)
    for result_file in config_and_files.files:
        result_files_by_mode[result_file.execution_mode].append(result_file)
    n_splits = config_and_files.config.number_of_cross_validation_splits
    failing_modes = []
    for mode, files in result_files_by_mode.items():
        if len(files) != n_splits:
            failing_modes.append(mode)
    if not failing_modes:
        return
    logging.warning(f"The expected number of runs to evaluate was {n_splits}.")
    for mode in failing_modes:
        files_for_mode = result_files_by_mode[mode]
        logging.warning(f"Number of result files of mode {mode} found is {len(files_for_mode)}, as follows:")
        for file in files_for_mode:
            logging.warning(f"  Metrics: {file.metrics_file} and dataset: {file.dataset_csv_file}")
    mode_string = ' '.join(str(mode) for mode in failing_modes)
    raise ValueError(f"Unexpected number(s) of runs to evaluate for mode(s) {mode_string}")


def plot_cross_validation_from_files(config_and_files: OfflineCrossvalConfigAndFiles,
                                     root_folder: Path) -> None:
    config = config_and_files.config
    if config.number_of_cross_validation_splits > 1:
        check_result_file_counts(config_and_files)
    result_files = config_and_files.files
    metrics_dfs = load_dataframes(result_files, config)
    full_csv_file = root_folder / FULL_METRICS_DATAFRAME_FILE
    initial_metrics = pd.concat(list(metrics_dfs.values()))
    if config.model_category == ModelCategory.Segmentation:
        if config.create_plots:
            plot_metrics(config, metrics_dfs, root_folder)
        save_outliers(config, metrics_dfs, root_folder)
        all_metrics, focus_splits = add_comparison_data(config, initial_metrics)
        all_metrics.to_csv(full_csv_file, index=False)
        run_statistical_tests_on_file(root_folder, full_csv_file, config, focus_splits)
    else:
        # For classification runs, we also want to compute the aggregated training metrics for
        # each fold.
        metrics = ScalarMetricsDict.load_execution_mode_metrics_from_df(
            initial_metrics,
            config.model_category == ModelCategory.Classification)
        ScalarMetricsDict.aggregate_and_save_execution_mode_metrics(
            metrics=metrics,
            data_frame_logger=DataframeLogger(
                csv_path=root_folder / METRICS_AGGREGATES_FILE
            )
        )
        # The full metrics file saves the prediction for each individual subject. Do not include the training
        # results in this file (as in cross-validation a subject is used in several folds.)
        val_and_test_metrics = initial_metrics.loc[
            initial_metrics[LoggingColumns.DataSplit.value] != ModelExecutionMode.TRAIN.value]
        val_and_test_metrics.to_csv(full_csv_file, index=False)

        # Copy one instance of the dataset.CSV files to the root of the results folder. It is possible
        # that the different CV folds run with different dataset files, but not expected for classification
        # models at the moment (could change with ensemble models)
        dataset_csv = None
        for file in result_files:
            if file.dataset_csv_file:
                dataset_csv = file.dataset_csv_file
                break
        if dataset_csv:
            shutil.copy(str(dataset_csv), str(root_folder))
    name_dct = config_and_files.config.short_names
    if name_dct:
        pairs = [(val, key) for key, val in name_dct.items()]
        with Path(root_folder / RUN_DICTIONARY_NAME).open("w") as out:
            max_len = max(len(short_name) for short_name, _ in pairs)
            for short_name, long_name in sorted(pairs):
                out.write(f"{short_name:{max_len}s}    {long_name}\n")


def get_metrics_columns(df: pd.DataFrame) -> Set[str]:
    """
    Get all columns of the data frame that appear to be metrics. A column is considered a metric if it appears
    as a dictionary value in INTERNAL_TO_LOGGING_COLUMN_NAMES.
    :param df: A dataframe to analyze.
    :return: The set of data frame columns that is also contained in INTERNAL_TO_LOGGING_COLUMN_NAMES.
    """
    all_metrics = set(v.value for v in INTERNAL_TO_LOGGING_COLUMN_NAMES.values())
    return set(df).intersection(all_metrics)


def unroll_aggregate_metrics(df: pd.DataFrame) -> List[EpochMetricValues]:
    """
    Consumes a dataframe that is read from an aggregate metrics file for classification or segmentation,
    and converts all entries for execution mode "Val" to (metric_name, metric_value) pairs, sorted by
    epoch ascendingly.
    :param df:
    :return:
    """
    if LoggingColumns.DataSplit.value not in df:
        raise ValueError(f"Column {LoggingColumns.DataSplit.value} must be present.")
    if LoggingColumns.Epoch.value not in df:
        raise ValueError(f"Column {LoggingColumns.Epoch.value} must be present.")
    metrics = sorted(get_metrics_columns(df))
    df.sort_values(by=LoggingColumns.Epoch.value, inplace=True)
    df = df[df[LoggingColumns.DataSplit.value] == ModelExecutionMode.VAL.value]
    result: List[EpochMetricValues] = []
    for epoch in sorted(df[LoggingColumns.Epoch.value].unique()):
        rows = df[df[LoggingColumns.Epoch.value] == epoch]
        assert len(rows) == 1, "Expected only a single row of aggregate metrics per epoch."
        d = rows.head(n=1).to_dict('records')[0]
        for metric in metrics:
            result.append(EpochMetricValues(epoch=epoch, metric_name=metric, metric_value=d[metric]))
    return result


def plot_cross_validation(config: PlotCrossValidationConfig) -> Path:
    """
    Collects results from an AzureML cross validation run, and writes aggregate metrics files.
    :param config: The settings for plotting cross validation results.
    :return:
    """
    logging_to_stdout(logging.INFO)
    with logging_section("Downloading cross-validation results"):
        result_files, root_folder = download_crossval_result_files(config)
    config_and_files = OfflineCrossvalConfigAndFiles(config=config, files=result_files)
    with logging_section("Plotting cross-validation results"):
        plot_cross_validation_from_files(config_and_files, root_folder)
    return root_folder


def add_comparison_data(config: PlotCrossValidationConfig, metrics: pd.DataFrame) \
        -> Tuple[pd.DataFrame, Optional[List[Any]]]:
    """
    :param config: configuration of this plotting run
    :param metrics: on entry, metrics for just the focus (target) run
    :return: if there are comparison runs, an extended version of metrics including statistics for those runs,
    and a list of focus (target) split names to compare them against; otherwise, "metrics" is returned, with
    focus_splits as None, so we will get an all-against-all statistical comparison.
    """
    if config.comparison_run_recovery_ids is None:
        return metrics, None
    focus_splits = list(metrics[COL_SPLIT].unique())
    for comparison_id, comparison_epoch in zip(config.comparison_run_recovery_ids, config.comparison_epochs):
        files, _ = download_crossval_result_files(config, comparison_id, comparison_epoch)
        aux_metrics_df = load_dataframes(files, config)
        metrics = metrics.append(pd.concat(list(aux_metrics_df.values())))
    return metrics, focus_splits


def main() -> None:
    plot_cross_validation(PlotCrossValidationConfig.parse_args())


if __name__ == '__main__':
    main()

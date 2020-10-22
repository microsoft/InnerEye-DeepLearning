#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
import stopit
from azureml._base_sdk_common import user_agent
from azureml._restclient.constants import RunStatus
from azureml.core import Model, Run

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, ParserResult, SourceConfig
from InnerEye.Azure.azure_runner import create_runner_parser, parse_args_and_add_yaml_variables, \
    parse_arguments, submit_to_azureml
from InnerEye.Azure.azure_util import PARENT_RUN_CONTEXT, RUN_CONTEXT, RUN_RECOVERY_ID_KEY_NAME
from InnerEye.Azure.run_pytest import download_pytest_result, run_pytest
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import BASELINE_COMPARISONS_FOLDER, BASELINE_WILCOXON_RESULTS_FILE, \
    CROSSVAL_RESULTS_FOLDER, ENSEMBLE_SPLIT_NAME, FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE, \
    METRICS_FILE_NAME, ModelProcessing, OTHER_RUNS_SUBDIR_NAME, SCATTERPLOTS_SUBDIR_NAME, disable_logging_to_file, \
    get_epoch_results_path, is_linux, logging_section, logging_to_file, logging_to_stdout, \
    print_exception, remove_file_or_directory
from InnerEye.Common.fixed_paths import get_environment_yaml_file
from InnerEye.ML.common import DATASET_CSV_FILE_NAME, ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.deep_learning_config import DeepLearningConfig, ModelCategory
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.reports.notebook_report import generate_segmentation_notebook, generate_classification_notebook
from InnerEye.ML.utils.config_util import ModelConfigLoader

REPORT_IPYNB = "report.ipynb"
REPORT_HTML = "report.html"

LOG_FILE_NAME = "stdout.txt"

PostCrossValidationHookSignature = Callable[[ModelConfigBase, Path], None]
ModelDeploymentHookSignature = Callable[[SegmentationModelBase, AzureConfig, Model, ModelProcessing],
                                        Tuple[Optional[Path], Optional[Any]]]


def may_initialize_rpdb() -> None:
    """
    On Linux only, import and initialize rpdb, to enable remote debugging if necessary.
    """
    # rpdb signal trapping does not work on Windows, as there is no SIGTRAP:
    if not is_linux():
        return
    import rpdb
    rpdb_port = 4444
    rpdb.handle_trap(port=rpdb_port)
    # For some reason, os.getpid() does not return the ID of what appears to be the currently running process.
    logging.info("rpdb is handling traps. To debug: identify the main runner.py process, then as root: "
                 f"kill -TRAP <process_id>; nc 127.0.0.1 {rpdb_port}")


def suppress_logging_noise() -> None:
    """
    Reduce the logging level for some of the used libraries, which are particularly talkative in DEBUG mode.
    Usually when running in DEBUG mode, we want diagnostics about the model building itself, but not for the
    underlying libraries.
    """
    # Numba code generation is extremely talkative in DEBUG mode, disable that.
    logging.getLogger('numba').setLevel(logging.WARNING)
    # Matplotlib is also very talkative in DEBUG mode, filling half of the log file in a PR build.
    logging.getLogger('matplotlib').setLevel(logging.INFO)
    # Urllib3 prints out connection information for each call to write metrics, etc
    logging.getLogger('urllib3').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)
    # Jupyter notebook report generation
    logging.getLogger('papermill').setLevel(logging.INFO)
    logging.getLogger('nbconvert').setLevel(logging.INFO)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


class Runner:
    """
    :param project_root: The root folder that contains all of the source code that should be executed.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    :param post_cross_validation_hook: A function to call after waiting for completion of cross validation runs.
    The function is called with the model configuration and the path to the downloaded and merged metrics files.
    :param model_deployment_hook: an optional function for deploying a model in an application-specific way.
    If present, it should take a model config (SegmentationModelBase), an AzureConfig, and an AzureML
    Model as arguments, and return an optional Path and a further object of any type.
    :param innereye_submodule_name: name of the InnerEye submodule if any; should be at top level.
    Suggested value is "innereye-deeplearning".
    :param command_line_args: command-line arguments to use; if None, use sys.argv.
    """

    def __init__(self,
                 project_root: Path,
                 yaml_config_file: Path,
                 post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
                 innereye_submodule_name: Optional[str] = None,
                 command_line_args: Optional[List[str]] = None):
        self.project_root = project_root
        self.yaml_config_file = yaml_config_file
        self.post_cross_validation_hook = post_cross_validation_hook
        self.model_deployment_hook = model_deployment_hook
        self.innereye_submodule_name = innereye_submodule_name
        self.command_line_args = command_line_args
        # model_config and azure_config are placeholders for now, and are set properly when command line args are
        # parsed.
        self.model_config: ModelConfigBase = ModelConfigBase(azure_dataset_id="")
        self.azure_config: AzureConfig = AzureConfig()

    def wait_until_cross_val_splits_are_ready_for_aggregation(self) -> bool:
        """
        Checks if all child runs (except the current run) of the current run's parent are completed or failed.
        If this is the case, then we can aggregate the results of the other runs before terminating this run.
        :return: whether we need to wait, i.e. whether some runs are still pending.
        """
        if (not self.model_config.is_offline_run) \
                and (azure_util.is_cross_validation_child_run(RUN_CONTEXT)):
            n_splits = self.model_config.get_total_number_of_cross_validation_runs()
            child_runs = azure_util.fetch_child_runs(PARENT_RUN_CONTEXT,
                                                     expected_number_cross_validation_splits=n_splits)
            pending_runs = [x.id for x in child_runs
                            if (x.id != RUN_CONTEXT.id)
                            and (x.get_status() not in [RunStatus.COMPLETED, RunStatus.FAILED])]
            should_wait = len(pending_runs) > 0
            if should_wait:
                logging.info(f"Waiting for sibling run(s) to finish: {pending_runs}")
            return should_wait
        else:
            raise NotImplementedError("cross_val_splits_are_ready_for_aggregation is implemented for online "
                                      "cross validation runs only")

    @stopit.threading_timeoutable()
    def wait_for_cross_val_runs_to_finish_and_aggregate(self, delay: int = 60) -> None:
        """
        Wait for cross val runs (apart from the current one) to finish and then aggregate results of all.
        :param delay: How long to wait between polls to AML to get status of child runs
        """
        with logging_section("Waiting for sibling runs"):
            while self.wait_until_cross_val_splits_are_ready_for_aggregation():
                time.sleep(delay)
        assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run"
        self.create_ensemble_model()

    @staticmethod
    def generate_report(config: DeepLearningConfig, best_epoch: int, model_proc: ModelProcessing) -> None:
        logging.info("Saving report in html")
        if config.model_category not in [ModelCategory.Segmentation, ModelCategory.Classification]:
            return

        try:
            def get_epoch_path(mode: ModelExecutionMode) -> Path:
                p = get_epoch_results_path(best_epoch, mode=mode, model_proc=model_proc)
                return config.outputs_folder / p / METRICS_FILE_NAME

            path_to_best_epoch_train = get_epoch_path(ModelExecutionMode.TRAIN)
            path_to_best_epoch_val = get_epoch_path(ModelExecutionMode.VAL)
            path_to_best_epoch_test = get_epoch_path(ModelExecutionMode.TEST)

            output_dir = config.outputs_folder / OTHER_RUNS_SUBDIR_NAME / ENSEMBLE_SPLIT_NAME \
                if model_proc == ModelProcessing.ENSEMBLE_CREATION else config.outputs_folder
            if config.model_category == ModelCategory.Segmentation:
                generate_segmentation_notebook(result_notebook=output_dir / REPORT_IPYNB,
                                               train_metrics=path_to_best_epoch_train,
                                               val_metrics=path_to_best_epoch_val,
                                               test_metrics=path_to_best_epoch_test)
            else:
                if isinstance(config, ScalarModelBase):
                    generate_classification_notebook(result_notebook=output_dir / REPORT_IPYNB,
                                                     train_metrics=path_to_best_epoch_train,
                                                     val_metrics=path_to_best_epoch_val,
                                                     test_metrics=path_to_best_epoch_test,
                                                     dataset_csv_path=config.local_dataset / DATASET_CSV_FILE_NAME
                                                                        if config.local_dataset else None,
                                                     dataset_subject_column=config.subject_column,
                                                     dataset_file_column=config.image_file_column)
                else:
                    logging.info(f"Cannot create report for config of type {type(config)}.")
        except Exception as ex:
            print_exception(ex, "Failed to generated reporting notebook.")

    def plot_cross_validation_and_upload_results(self) -> Path:
        from InnerEye.ML.visualizers.plot_cross_validation import crossval_config_from_model_config, \
            plot_cross_validation, unroll_aggregate_metrics
        # perform aggregation as cross val splits are now ready
        plot_crossval_config = crossval_config_from_model_config(self.model_config)
        plot_crossval_config.run_recovery_id = PARENT_RUN_CONTEXT.tags[RUN_RECOVERY_ID_KEY_NAME]
        plot_crossval_config.outputs_directory = str(self.model_config.outputs_folder)
        plot_crossval_config.settings_yaml_file = str(self.yaml_config_file)
        cross_val_results_root = plot_cross_validation(plot_crossval_config)
        if self.post_cross_validation_hook:
            self.post_cross_validation_hook(self.model_config, cross_val_results_root)
        # upload results to the parent run's outputs. Normally, we use blobxfer for that, but here we want
        # to ensure that the files are visible inside the AzureML UI.
        PARENT_RUN_CONTEXT.upload_folder(name=CROSSVAL_RESULTS_FOLDER, path=str(cross_val_results_root))
        if self.model_config.is_scalar_model:
            try:
                aggregates = pd.read_csv(cross_val_results_root / METRICS_AGGREGATES_FILE)
                unrolled_aggregate_metrics = unroll_aggregate_metrics(aggregates)
                for m in unrolled_aggregate_metrics:
                    PARENT_RUN_CONTEXT.log(m.metric_name, m.metric_value)
            except Exception as ex:
                print_exception(ex, "Unable to log metrics to Hyperdrive parent run.", logger_fn=logging.warning)
        return cross_val_results_root

    def create_ensemble_model(self) -> None:
        """
        Call MLRunner again after training cross-validation models, to create an ensemble model from them.
        """
        # Import only here in case of dependency issues in reduced environment
        from InnerEye.ML.utils.run_recovery import RunRecovery
        with logging_section("Downloading checkpoints from sibling runs"):
            run_recovery = RunRecovery.download_checkpoints_from_run(
                self.azure_config, self.model_config, PARENT_RUN_CONTEXT, output_subdir_name=OTHER_RUNS_SUBDIR_NAME)
            # Check paths are good, just in case
            for path in run_recovery.checkpoints_roots:
                if not path.is_dir():
                    raise NotADirectoryError(f"Does not exist or is not a directory: {path}")
        # Adjust parameters
        self.azure_config.hyperdrive = False
        self.model_config.number_of_cross_validation_splits = 0
        self.model_config.is_train = False
        best_epoch = self.create_ml_runner().run_inference_and_register_model(run_recovery,
                                                                              model_proc=ModelProcessing.ENSEMBLE_CREATION)

        crossval_dir = self.plot_cross_validation_and_upload_results()
        Runner.generate_report(self.model_config, best_epoch, ModelProcessing.ENSEMBLE_CREATION)
        # CrossValResults should have been uploaded to the parent run, so we don't need it here.
        remove_file_or_directory(crossval_dir)
        # We can also remove OTHER_RUNS under the root, as it is no longer useful and only contains copies of files
        # available elsewhere. However, first we need to upload relevant parts of OTHER_RUNS/ENSEMBLE.
        other_runs_dir = self.model_config.outputs_folder / OTHER_RUNS_SUBDIR_NAME
        other_runs_ensemble_dir = other_runs_dir / ENSEMBLE_SPLIT_NAME
        if PARENT_RUN_CONTEXT is not None:
            if other_runs_ensemble_dir.exists():
                # Only keep baseline Wilcoxon results and scatterplots and reports
                for subdir in other_runs_ensemble_dir.glob("*"):
                    if subdir.name not in [BASELINE_WILCOXON_RESULTS_FILE, SCATTERPLOTS_SUBDIR_NAME, REPORT_HTML,
                                           REPORT_IPYNB]:
                        remove_file_or_directory(subdir)
                PARENT_RUN_CONTEXT.upload_folder(name=BASELINE_COMPARISONS_FOLDER, path=str(other_runs_ensemble_dir))
            else:
                logging.warning(f"Directory not found for upload: {other_runs_ensemble_dir}")
        remove_file_or_directory(other_runs_dir)

    def parse_and_load_model(self) -> ParserResult:
        """
        Parses the command line arguments, and creates configuration objects for the model itself, and for the
        Azure-related parameters. Sets self.azure_config and self.model_config to their proper values.
        """
        # Create a parser that will understand only the args we need for an AzureConfig
        parser1 = create_runner_parser()
        parser1_result = parse_args_and_add_yaml_variables(parser1,
                                                           yaml_config_file=self.yaml_config_file,
                                                           project_root=self.project_root,
                                                           args=self.command_line_args,
                                                           fail_on_unknown_args=False)
        azure_config = AzureConfig(**parser1_result.args)
        azure_config.project_root = self.project_root
        model_config_loader: ModelConfigLoader = ModelConfigLoader(**parser1_result.args)
        # Create the model as per the "model" commandline option
        model_config = model_config_loader.create_model_config_from_name(
            model_name=azure_config.model
        )
        # This model will be either a classification model or a segmentation model. Those have different
        # fields that could be overridden on the command line. Create a parser that understands the fields we need
        # for the actual model type. We feed this parser will the YAML settings and commandline arguments that the
        # first parser did not recognize.
        parser2 = type(model_config).create_argparser()
        parser2_result = parse_arguments(parser2,
                                         settings_from_yaml=parser1_result.unknown_settings_from_yaml,
                                         args=parser1_result.unknown,
                                         fail_on_unknown_args=True)
        # Apply the overrides and validate. Overrides can come from either YAML settings or the commandline.
        model_config.apply_overrides(parser1_result.unknown_settings_from_yaml)
        model_config.apply_overrides(parser2_result.overrides)
        model_config.validate()
        # Set the file system related configs, they might be affected by the overrides that were applied.
        logging.info("Creating the adjusted output folder structure.")
        model_config.create_filesystem(self.project_root)
        if azure_config.extra_code_directory:
            exist = "exists" if Path(azure_config.extra_code_directory).exists() else "does not exist"
            logging.info(f"extra_code_directory is {azure_config.extra_code_directory}, which {exist}")
        else:
            logging.info("extra_code_directory is unset")
        self.azure_config = azure_config
        self.model_config = model_config
        return parser2_result

    def run(self) -> Tuple[ModelConfigBase, Optional[Run]]:
        """
        The main entry point for training and testing models from the commandline. This chooses a model to train
        via a commandline argument, runs training or testing, and writes all required info to disk and logs.
        :return: If submitting to AzureML, returns the model configuration that was used for training,
        including commandline overrides applied (if any).
        """
        # Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(logging.INFO)
        may_initialize_rpdb()
        user_agent.append(azure_util.INNEREYE_SDK_NAME, azure_util.INNEREYE_SDK_VERSION)
        self.parse_and_load_model()
        if self.model_config.perform_cross_validation:
            # force hyperdrive usage if performing cross validation
            self.azure_config.hyperdrive = True
        run_object: Optional[Run] = None
        if self.azure_config.azureml:
            run_object = self.submit_to_azureml()
        else:
            self.run_in_situ()
        return self.model_config, run_object

    def submit_to_azureml(self) -> Run:
        """
        Submit a job to AzureML, returning the resulting Run object, or exiting if we were asked to wait for
        completion and the Run did not succeed.
        """
        # The adal package creates a logging.info line each time it gets an authentication token, avoid that.
        logging.getLogger('adal-python').setLevel(logging.WARNING)
        if not self.model_config.azure_dataset_id:
            raise ValueError("When running on AzureML, the 'azure_dataset_id' property must be set.")
        model_config_overrides = str(self.model_config.overrides)
        source_config = SourceConfig(
            root_folder=str(self.project_root),
            entry_script=os.path.abspath(sys.argv[0]),

            conda_dependencies_files=[get_environment_yaml_file(),
                                      self.project_root / fixed_paths.ENVIRONMENT_YAML_FILE_NAME],
            hyperdrive_config_func=lambda estimator: self.model_config.get_hyperdrive_config(estimator),
            # For large jobs, upload of results times out frequently because of large checkpoint files. Default is 600
            upload_timeout_seconds=86400,
        )
        source_config.set_script_params_except_submit_flag()
        assert self.model_config.azure_dataset_id is not None  # to stop mypy complaining about next line
        azure_run = submit_to_azureml(self.azure_config, source_config, model_config_overrides,
                                      self.model_config.azure_dataset_id)
        logging.info("Job submission to AzureML done.")
        if self.azure_config.pytest_mark:
            # The AzureML job can optionally run pytest. Attempt to download it to the current directory.
            # A build step will pick up that file and publish it to Azure DevOps.
            # If pytest_mark is set, this file must exist.
            logging.info("Downloading pytest result file.")
            download_pytest_result(azure_run)
        else:
            logging.info("No pytest_mark present, hence not downloading the pytest result file.")
        status = azure_run.get_status()
        # For PR builds where we wait for job completion, the job must have ended in a COMPLETED state.
        # If a pytest failed, the runner has exited with code -1 (see below)
        if self.azure_config.wait_for_completion and status != RunStatus.COMPLETED:
            logging.error(f"Job completed with status {status}. Exiting.")
            exit(-1)
        return azure_run

    def run_in_situ(self) -> None:
        """
        Actually run the AzureML job; this method will typically run on an Azure VM.
        """
        # Only set the logging level now. Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(self.azure_config.log_level)
        suppress_logging_noise()
        pytest_failed = False
        training_failed = False
        pytest_passed = True
        # Ensure that both model training and pytest both get executed in all cases, so that we see a full set of
        # test results in each PR
        outputs_folder = self.model_config.outputs_folder
        try:
            logging_to_file(self.model_config.logs_folder / LOG_FILE_NAME)
            try:
                self.create_ml_runner().run()
            except Exception as ex:
                print_exception(ex, "Model training/testing failed.")
                training_failed = True
            if self.azure_config.pytest_mark:
                try:
                    pytest_passed, results_file_path = run_pytest(self.azure_config.pytest_mark, outputs_folder)
                    if not pytest_passed:
                        logging.error(
                            f"Not all PyTest tests passed. See {results_file_path}")
                except Exception as ex:
                    print_exception(ex, "Unable to run PyTest.")
                    pytest_failed = True
        finally:
            # wait for aggregation if required, and only if the training actually succeeded.
            if not training_failed and self.model_config.should_wait_for_other_cross_val_child_runs():
                self.wait_for_cross_val_runs_to_finish_and_aggregate()
            disable_logging_to_file()
        if training_failed or pytest_failed or not pytest_passed:
            # Terminate if pytest or model training has failed. This makes the smoke test in
            # PR builds fail if pytest fails.
            exit(-1)

    def create_ml_runner(self) -> Any:
        """
        Create and return an ML runner using the attributes of this Runner object.
        """
        # This import statement cannot be at the beginning of the file because it will cause import
        # of packages that are not available inside the azure_runner.yml environment: torch, blobxfer.
        # That is also why we specify the return type as Any rather than MLRunner.
        from InnerEye.ML.run_ml import MLRunner
        return MLRunner(
            self.model_config, self.azure_config, self.project_root,
            self.model_deployment_hook, self.innereye_submodule_name)


def default_post_cross_validation_hook(config: ModelConfigBase, root_folder: Path) -> None:
    """
    A function to run after cross validation results have been aggregated, before they are uploaded to AzureML.
    :param config: The configuration of the model that should be trained.
    :param root_folder: The folder with all aggregated and per-split files.
    """
    print(f"Analyzing cross validation results for model {config.model_name}")
    print(f"Expecting all cross validation result files in folder {root_folder}")
    for (file, description) in [
        (DATASET_CSV_FILE_NAME, "Dataset"),
        (METRICS_AGGREGATES_FILE, "Metrics aggregated at epoch level"),
        (FULL_METRICS_DATAFRAME_FILE, "Metrics at subject level")
    ]:
        full_file = root_folder / file
        print(f"{description} (exists={full_file.exists()}): {full_file}")


def run(project_root: Path,
        yaml_config_file: Path,
        post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
        model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
        innereye_submodule_name: Optional[str] = None,
        command_line_args: Optional[List[str]] = None) -> \
        Tuple[ModelConfigBase, Optional[Run]]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    runner = Runner(project_root, yaml_config_file, post_cross_validation_hook,
                    model_deployment_hook, innereye_submodule_name, command_line_args)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory(),
        yaml_config_file=fixed_paths.SETTINGS_YAML_FILE,
        post_cross_validation_hook=default_post_cross_validation_hook)


if __name__ == '__main__':
    main()

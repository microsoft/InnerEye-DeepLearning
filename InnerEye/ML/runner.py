#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

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
from InnerEye.Common.common_util import CROSSVAL_RESULTS_FOLDER, FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE, \
    disable_logging_to_file, is_linux, logging_to_file, logging_to_stdout, print_exception
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils.config_util import ModelConfigLoader

LOG_FILE_NAME = "stdout.txt"

PostCrossValidationHookSignature = Callable[[ModelConfigBase, Path], None]
ModelDeploymentHookSignature = Callable[[SegmentationModelBase, AzureConfig, Model],
                                        Tuple[Optional[Path], Optional[Any]]]


def wait_until_cross_val_splits_are_ready_for_aggregation(model_config: ModelConfigBase) -> bool:
    """
    Checks if all child runs (except the current run) of the current run's parent are completed or failed.
    If this is the case, then we can aggregate the results of the other runs before terminating this run.
    :return:
    """
    if (not model_config.is_offline_run) \
            and (azure_util.is_cross_validation_child_run(RUN_CONTEXT)):
        pending_runs = [x.id for x in azure_util.fetch_child_runs(PARENT_RUN_CONTEXT,
                                                                  expected_number_cross_validation_splits=model_config.number_of_cross_validation_splits)
                        if (x.id != RUN_CONTEXT.id)
                        and (x.get_status() not in [RunStatus.COMPLETED, RunStatus.FAILED])]
        should_wait = len(pending_runs) > 0
        if should_wait:
            logging.info(f"Waiting for child runs to finish: {pending_runs}")
        return should_wait
    else:
        raise NotImplementedError("cross_val_splits_are_ready_for_aggregation is implemented for online "
                                  "cross validation runs only")


@stopit.threading_timeoutable()
def wait_for_cross_val_runs_to_finish_and_aggregate(
        model_config: ModelConfigBase,
        train_yaml_path: Path,
        post_cross_validation_hook: Optional[PostCrossValidationHookSignature],
        delay: int = 60) -> Path:
    """
    Wait for cross val runs (apart from the current one) to finish and then aggregate results of all.
    :param model_config: Model related configurations.
    :param train_yaml_path: YAML Config file for training variables.
    :param post_cross_validation_hook: A function to call after waiting for completion of cross validation runs.
    The function is called with the path to the downloaded and merged metrics files.
    :param delay: How long to wait between polls to AML to get status of child runs
    :return: Path to the aggregated results.
    """
    from InnerEye.ML.visualizers.plot_cross_validation import crossval_config_from_model_config, \
        plot_cross_validation, unroll_aggregate_metrics
    while wait_until_cross_val_splits_are_ready_for_aggregation(model_config):
        time.sleep(delay)

    assert PARENT_RUN_CONTEXT, "This function should only be called in a Hyperdrive run"
    # perform aggregation as cross val splits are now ready
    plot_crossval_config = crossval_config_from_model_config(model_config)
    plot_crossval_config.run_recovery_id = PARENT_RUN_CONTEXT.tags[RUN_RECOVERY_ID_KEY_NAME]
    plot_crossval_config.outputs_directory = str(model_config.outputs_folder)
    plot_crossval_config.train_yaml_path = str(train_yaml_path)
    cross_val_results_root = plot_cross_validation(plot_crossval_config)
    if post_cross_validation_hook:
        post_cross_validation_hook(model_config, cross_val_results_root)
    # upload results to the parent run's outputs. Normally, we use blobxfer for that, but here we want
    # to ensure that the files are visible inside the AzureML UI.
    PARENT_RUN_CONTEXT.upload_folder(name=CROSSVAL_RESULTS_FOLDER, path=str(cross_val_results_root))
    if model_config.is_scalar_model:
        try:
            aggregates = pd.read_csv(cross_val_results_root / METRICS_AGGREGATES_FILE)
            unrolled_aggregate_metrics = unroll_aggregate_metrics(aggregates)
            for m in unrolled_aggregate_metrics:
                PARENT_RUN_CONTEXT.log(m.metric_name, m.metric_value)
        except Exception as ex:
            print_exception(ex, "Unable to log metrics to Hyperdrive parent run.", logger_fn=logging.warning)
    return cross_val_results_root


@dataclass(frozen=True)
class ConfigurationsAndParserResults:
    """
    Contains the parsed model configuration and the Azure-related configuration.
    """
    model_config: ModelConfigBase
    azure_config: AzureConfig
    parser_result: ParserResult


def parse_and_load_model(project_root: Path, yaml_config_file: Path) -> ConfigurationsAndParserResults:
    """
    Parses the command line arguments, and creates configuration objects for the model itself, and for the
    Azure-related parameters.
    :param project_root: The root folder that contains all of the source code that should be executed.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    If not supplied, use train.yaml at the repository root.
    :return:
    """
    # Create a parser that will understand only the args we need for an AzureConfig
    parser1 = create_runner_parser()
    parser1_result = parse_args_and_add_yaml_variables(parser1,
                                                       yaml_config_file=yaml_config_file,
                                                       fail_on_unknown_args=False)
    azure_config = AzureConfig(**parser1_result.args)
    azure_config.project_root = project_root
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
    model_config.create_filesystem(project_root)
    if azure_config.extra_code_directory:
        exist = "exists" if Path(azure_config.extra_code_directory).exists() else "does not exist"
        logging.info(f"extra_code_directory is {azure_config.extra_code_directory}, which {exist}")
    else:
        logging.info(f"extra_code_directory is unset")
    return ConfigurationsAndParserResults(
        model_config=model_config,
        azure_config=azure_config,
        parser_result=parser2_result
    )


def run(project_root: Path,
        yaml_config_file: Path,
        post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
        model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
        innereye_submodule_name: Optional[str] = None) -> \
        Tuple[ModelConfigBase, Optional[Run]]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.
    :param post_cross_validation_hook: A function to call after waiting for completion of cross validation runs.
    The function is called with the model configuration and the path to the downloaded and merged metrics files.
    :param project_root: The root folder that contains all of the source code that should be executed.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    :param model_deployment_hook: an optional function for deploying a model in an application-specific way.
    If present, it should take a model config (SegmentationModelBase), an AzureConfig, and an AzureML
    :param innereye_submodule_name: submodule name
    Model as arguments, and return an optional Path and a further object of any type.
    :param innereye_submodule_name: name of the InnerEye submodule if any; should be at top level.
    Suggested value is "innereye-deeplearning".
    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any).
    """
    # Usually, when we set logging to DEBUG, we want diagnostics about the model
    # build itself, but not the tons of debug information that AzureML submissions create.
    logging_to_stdout(logging.INFO)
    # rpdb signal trapping does not work on Windows, as there is no SIGTRAP:
    if is_linux():
        import rpdb
        rpdb_port = 4444
        rpdb.handle_trap(port=rpdb_port)
        # For some reason, os.getpid() does not return the ID of what appears to be the currently running process.
        logging.info(f"rpdb is handling traps. To debug: identify the main runner.py process, then as root: "
                     f"kill -TRAP <process_id>; nc 127.0.0.1 {rpdb_port}")
    user_agent.append(azure_util.INNEREYE_SDK_NAME, azure_util.INNEREYE_SDK_VERSION)
    parse_and_load_result = parse_and_load_model(project_root, yaml_config_file=yaml_config_file)
    model_config = parse_and_load_result.model_config
    azure_config = parse_and_load_result.azure_config
    if model_config.number_of_cross_validation_splits > 0:
        # force hyperdrive usage if performing cross validation
        azure_config.hyperdrive = True
    if azure_config.submit_to_azureml:
        # The adal package creates a logging.info line each time it gets an authentication token, avoid that.
        logging.getLogger('adal-python').setLevel(logging.WARNING)
        if not model_config.azure_dataset_id:
            raise ValueError(f"When running on AzureML, the 'azure_dataset_id' property must be set.")
        model_config_overrides = str(model_config.overrides)
        source_config = SourceConfig(
            root_folder=str(project_root),
            entry_script=os.path.abspath(sys.argv[0]),
            conda_dependencies_file=project_root / fixed_paths.ENVIRONMENT_YAML_FILE_NAME,
            hyperdrive_config_func=lambda estimator: model_config.get_hyperdrive_config(estimator),
            # For large jobs, upload of results times out frequently because of large checkpoint files. Default is 600
            upload_timeout_seconds=86400,
        )
        source_config.set_script_params_except_submit_flag()
        assert model_config.azure_dataset_id is not None  # to stop mypy complaining about next line
        azure_run = submit_to_azureml(azure_config, source_config, model_config_overrides,
                                      model_config.azure_dataset_id)
        logging.info("Job submission to AzureML done.")
        if azure_config.pytest_mark:
            # The AzureML job can optionally run pytest. Attempt to download it to the current directory.
            # A build step will pick up that file and publish it to Azure DevOps.
            # If pytest_mark is set, this file must exist.
            logging.info(f"Downloading pytest result file.")
            download_pytest_result(azure_config, azure_run)
        else:
            logging.info("No pytest_mark present, hence not downloading the pytest result file.")
        status = azure_run.get_status()
        # For PR builds where we wait for job completion, the job must have ended in a COMPLETED state.
        # If a pytest failed, the runner has exited with code -1 (see below)
        if azure_config.wait_for_completion and status != RunStatus.COMPLETED:
            logging.error(f"Job completed with status {status}. Exiting.")
            exit(-1)
        return model_config, azure_run
    else:
        # This import statement cannot be at the beginning of the file because it will cause import
        # of packages that are not available inside the azure_runner.yml environment: torch, blobxfer
        from InnerEye.ML.run_ml import MLRunner
        # Only set the logging level now. Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(azure_config.log_level)
        # Numba code generation is extremely talkative in DEBUG mode, disable that.
        logging.getLogger('numba').setLevel(logging.WARNING)
        pytest_failed = False
        training_failed = False
        pytest_passed = True
        # Ensure that both model training and pytest both get executed in all cases, so that we see a full set of
        # test results in each PR
        outputs_folder = model_config.outputs_folder
        try:
            logging_to_file(model_config.logs_folder / LOG_FILE_NAME)
            try:
                ml_runner = MLRunner(model_config, azure_config, project_root, model_deployment_hook,
                                     innereye_submodule_name)
                ml_runner.run()
            except Exception as ex:
                print_exception(ex, "Model training/testing failed.")
                training_failed = True
            if azure_config.pytest_mark:
                try:
                    pytest_passed, results_file_path = run_pytest(azure_config.pytest_mark, outputs_folder)
                    if not pytest_passed:
                        logging.error(
                            f"Not all PyTest tests passed. See {results_file_path}")
                except Exception as ex:
                    print_exception(ex, "Unable to run PyTest.")
                    pytest_failed = True
        finally:
            # wait for aggregation if required, and only if the training actually succeeded.
            if not training_failed and model_config.should_wait_for_other_cross_val_child_runs:
                wait_for_cross_val_runs_to_finish_and_aggregate(model_config,
                                                                yaml_config_file,
                                                                post_cross_validation_hook)
            disable_logging_to_file()
        if training_failed or pytest_failed or not pytest_passed:
            # Terminate if pytest or model training has failed. This makes the smoke test in PR builds fail if pytest
            # fails.
            exit(-1)
        return model_config, None


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


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory(),
        yaml_config_file=fixed_paths.TRAIN_YAML_FILE,
        post_cross_validation_hook=default_post_cross_validation_hook,
        model_deployment_hook=None)


if __name__ == '__main__':
    main()

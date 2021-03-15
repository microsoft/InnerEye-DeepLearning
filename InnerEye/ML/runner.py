#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import os
import sys
from pathlib import Path

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa

# Workaround for an issue with how AzureML and Pytorch Lightning interact: When spawning additional processes for DDP,
# the working directory is not correctly picked up in sys.path
print("Starting InnerEye runner.")
innereye_root = Path(__file__).absolute().parent.parent.parent
if (innereye_root / "InnerEye").is_dir():
    innereye_root_str = str(innereye_root)
    if innereye_root_str not in sys.path:
        print(f"Adding to sys.path: {innereye_root_str}")
        sys.path.insert(0, innereye_root_str)

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from azureml._base_sdk_common import user_agent
from azureml.core import Model, Run

from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, ParserResult, SourceConfig
from InnerEye.Azure.azure_runner import create_runner_parser, parse_args_and_add_yaml_variables, \
    parse_arguments, set_environment_variables_for_multi_node, submit_to_azureml
from InnerEye.Azure.azure_util import is_run_and_child_runs_completed
from InnerEye.Azure.run_pytest import download_pytest_result, run_pytest
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE, \
    ModelProcessing, disable_logging_to_file, is_linux, logging_to_file, logging_to_stdout, print_exception
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.utils.config_util import ModelConfigLoader

REPORT_IPYNB = "report.ipynb"
REPORT_HTML = "report.html"

LOG_FILE_NAME = "stdout.txt"

PostCrossValidationHookSignature = Callable[[ModelConfigBase, Path], None]
ModelDeploymentHookSignature = Callable[[SegmentationModelBase, AzureConfig, Model, ModelProcessing], Any]


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
    logging.getLogger('msrest').setLevel(logging.INFO)
    # AzureML prints too many details about logging metrics
    logging.getLogger('azureml').setLevel(logging.INFO)
    # Jupyter notebook report generation
    logging.getLogger('papermill').setLevel(logging.INFO)
    logging.getLogger('nbconvert').setLevel(logging.INFO)
    # This is working around a spurious error message thrown by MKL, see
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'


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


class Runner:
    """
    :param project_root: The root folder that contains all of the source code that should be executed.
    :param yaml_config_file: The path to the YAML file that contains values to supply into sys.argv.
    :param post_cross_validation_hook: A function to call after waiting for completion of cross validation runs.
    The function is called with the model configuration and the path to the downloaded and merged metrics files.
    :param model_deployment_hook: an optional function for deploying a model in an application-specific way.
    If present, it should take a model config (SegmentationModelBase), an AzureConfig, and an AzureML
    Model as arguments, and return an optional Path and a further object of any type.
    :param command_line_args: command-line arguments to use; if None, use sys.argv.
    """

    def __init__(self,
                 project_root: Path,
                 yaml_config_file: Path,
                 post_cross_validation_hook: Optional[PostCrossValidationHookSignature] = None,
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None,
                 command_line_args: Optional[List[str]] = None):
        self.project_root = project_root
        self.yaml_config_file = yaml_config_file
        self.post_cross_validation_hook = post_cross_validation_hook
        self.model_deployment_hook = model_deployment_hook
        self.command_line_args = command_line_args
        # model_config and azure_config are placeholders for now, and are set properly when command line args are
        # parsed.
        self.model_config: ModelConfigBase = ModelConfigBase(azure_dataset_id="")
        self.azure_config: AzureConfig = AzureConfig()

    def parse_and_load_model(self) -> Optional[ParserResult]:
        """
        Parses the command line arguments, and creates configuration objects for the model itself, and for the
        Azure-related parameters. Sets self.azure_config and self.model_config to their proper values. Returns the
        parser output from parsing the model commandline arguments.
        If no "model" argument is provided on the commandline, self.model_config will be set to None, and the return
        value is None.
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
        self.azure_config = azure_config
        self.model_config = None  # type: ignore
        if not azure_config.model:
            return None
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
        if self.model_config is not None and self.model_config.perform_cross_validation:
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
            root_folder=self.project_root,
            entry_script=Path(sys.argv[0]).resolve(),
            conda_dependencies_files=get_all_environment_files(self.project_root),
            hyperdrive_config_func=lambda run_config: self.model_config.get_hyperdrive_config(run_config),
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
        # For PR builds where we wait for job completion, the job must have ended in a COMPLETED state.
        if self.azure_config.wait_for_completion and not is_run_and_child_runs_completed(azure_run):
            raise ValueError(f"Run {azure_run.id} in experiment {azure_run.experiment.name} or one of its child "
                             "runs failed.")
        return azure_run

    def run_in_situ(self) -> None:
        """
        Actually run the AzureML job; this method will typically run on an Azure VM.
        """
        # Only set the logging level now. Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(self.azure_config.log_level)
        suppress_logging_noise()
        error_messages = []
        # For the PR build in AzureML, we can either pytest, or the training of the simple PR model. Running both
        # only works when using DDP_spawn, but that has as a side-effect that it messes up memory consumption of the
        # large models.
        if self.azure_config.pytest_mark:
            try:
                outputs_folder = Path.cwd() / fixed_paths.DEFAULT_AML_UPLOAD_DIR
                pytest_passed, results_file_path = run_pytest(self.azure_config.pytest_mark, outputs_folder)
                if not pytest_passed:
                    pytest_failures = f"Not all PyTest tests passed. See {results_file_path}"
                    logging.error(pytest_failures)
                    error_messages.append(pytest_failures)
            except Exception as ex:
                print_exception(ex, "Unable to run PyTest.")
                error_messages.append(f"Unable to run PyTest: {ex}")
        else:
            # Set environment variables for multi-node training if needed.
            # In particular, the multi-node environment variables should NOT be set in single node
            # training, otherwise this might lead to errors with the c10 distributed backend
            # (https://github.com/microsoft/InnerEye-DeepLearning/issues/395)
            if self.azure_config.num_nodes > 1:
                set_environment_variables_for_multi_node()
            try:
                logging_to_file(self.model_config.logs_folder / LOG_FILE_NAME)
                try:
                    self.create_ml_runner().run()
                except Exception as ex:
                    print_exception(ex, "Model training/testing failed.")
                    error_messages.append(f"Training failed: {ex}")
            finally:
                disable_logging_to_file()
        # Terminate if pytest or model training has failed. This makes the smoke test in
        # PR builds fail if pytest fails.
        if error_messages:
            raise ValueError(
                f"At least one component of the runner failed: {os.linesep} {os.linesep.join(error_messages)}")

    def create_ml_runner(self) -> Any:
        """
        Create and return an ML runner using the attributes of this Runner object.
        """
        # This import statement cannot be at the beginning of the file because it will cause import
        # of packages that are not available inside the azure_runner.yml environment, in particular pytorch.
        # That is also why we specify the return type as Any rather than MLRunner.
        from InnerEye.ML.run_ml import MLRunner
        return MLRunner(
            model_config=self.model_config,
            azure_config=self.azure_config,
            project_root=self.project_root,
            post_cross_validation_hook=self.post_cross_validation_hook,
            model_deployment_hook=self.model_deployment_hook)


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
        command_line_args: Optional[List[str]] = None) -> \
        Tuple[ModelConfigBase, Optional[Run]]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    runner = Runner(project_root, yaml_config_file, post_cross_validation_hook,
                    model_deployment_hook, command_line_args)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory(),
        yaml_config_file=fixed_paths.SETTINGS_YAML_FILE,
        post_cross_validation_hook=default_post_cross_validation_hook)


if __name__ == '__main__':
    main()

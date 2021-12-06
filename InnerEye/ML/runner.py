#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Optional, Tuple

# Suppress all errors here because the imports after code cause loads of warnings. We can't specifically suppress
# individual warnings only.
# flake8: noqa

# Workaround for an issue with how AzureML and Pytorch Lightning interact: When spawning additional processes for DDP,
# the working directory is not correctly picked up in sys.path
print(f"Starting InnerEye runner at {sys.argv[0]}")
innereye_root = Path(__file__).absolute().parent.parent.parent
if (innereye_root / "InnerEye").is_dir():
    innereye_root_str = str(innereye_root)
    if innereye_root_str not in sys.path:
        print(f"Adding InnerEye folder to sys.path: {innereye_root_str}")
        sys.path.insert(0, innereye_root_str)
from InnerEye.Common import fixed_paths

fixed_paths.add_submodules_to_path()

from azureml._base_sdk_common import user_agent
from azureml._restclient.constants import RunStatus
from azureml.core import Run, ScriptRunConfig
from health_azure import AzureRunInfo, submit_to_azure_if_needed
from health_azure.utils import create_run_recovery_id, is_global_rank_zero, is_local_rank_zero, merge_conda_files, \
    to_azure_friendly_string
import matplotlib

from InnerEye.Azure.tensorboard_monitor import AMLTensorBoardMonitorConfig, monitor
from InnerEye.Azure import azure_util
from InnerEye.Azure.azure_config import AzureConfig, ParserResult, SourceConfig
from InnerEye.Azure.azure_runner import (DEFAULT_DOCKER_BASE_IMAGE, create_dataset_configs, create_experiment_name,
                                         create_runner_parser,
                                         get_git_tags,
                                         parse_args_and_add_yaml_variables,
                                         parse_arguments, additional_run_tags,
                                         set_environment_variables_for_multi_node)
from InnerEye.Azure.azure_util import (RUN_CONTEXT, RUN_RECOVERY_ID_KEY_NAME, get_all_environment_files,
                                       is_offline_run_context)
from InnerEye.Azure.run_pytest import download_pytest_result, run_pytest
from InnerEye.Common.common_util import (FULL_METRICS_DATAFRAME_FILE, METRICS_AGGREGATES_FILE,
                                         is_linux, logging_to_stdout)
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.common import DATASET_CSV_FILE_NAME
from InnerEye.ML.deep_learning_config import DeepLearningConfig
from InnerEye.ML.lightning_base import InnerEyeContainer
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.run_ml import MLRunner, ModelDeploymentHookSignature, PostCrossValidationHookSignature
from InnerEye.ML.utils.config_loader import ModelConfigLoader
from InnerEye.ML.lightning_container import LightningContainer

# We change the current working directory before starting the actual training. However, this throws off starting
# the child training threads because sys.argv[0] is a relative path when running in AzureML. Turn that into an absolute
# path.
runner_path = Path(sys.argv[0])
if not runner_path.is_absolute():
    sys.argv[0] = str(runner_path.absolute())


def initialize_rpdb() -> None:
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


def package_setup_and_hacks() -> None:
    """
    Set up the Python packages where needed. In particular, reduce the logging level for some of the used
    libraries, which are particularly talkative in DEBUG mode. Usually when running in DEBUG mode, we want
    diagnostics about the model building itself, but not for the underlying libraries.
    It also adds workarounds for known issues in some packages.
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
    # Workaround for issues with matplotlib on some X servers, see
    # https://stackoverflow.com/questions/45993879/matplot-lib-fatal-io-error-25-inappropriate-ioctl-for-device-on-x
    # -server-loc
    matplotlib.use('Agg')


class Runner:
    """
    This class contains the high-level logic to start a training run: choose a model configuration by name,
    submit to AzureML if needed, or otherwise start the actual training and test loop.
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
                 model_deployment_hook: Optional[ModelDeploymentHookSignature] = None):
        self.project_root = project_root
        self.yaml_config_file = yaml_config_file
        self.post_cross_validation_hook = post_cross_validation_hook
        self.model_deployment_hook = model_deployment_hook
        # model_config and azure_config are placeholders for now, and are set properly when command line args are
        # parsed.
        self.model_config: Optional[DeepLearningConfig] = None
        self.azure_config: AzureConfig = AzureConfig()
        self.lightning_container: LightningContainer = None  # type: ignore
        # This field stores the MLRunner object that has been created in the most recent call to the run() method.
        self.ml_runner: Optional[MLRunner] = None

    def parse_and_load_model(self) -> ParserResult:
        """
        Parses the command line arguments, and creates configuration objects for the model itself, and for the
        Azure-related parameters. Sets self.azure_config and self.model_config to their proper values. Returns the
        parser output from parsing the model commandline arguments.
        If no "model" argument is provided on the commandline, self.model_config will be set to None, and the return
        value is None.
        """
        # Create a parser that will understand only the args we need for an AzureConfig
        parser1 = create_runner_parser()
        parser_result = parse_args_and_add_yaml_variables(parser1,
                                                          yaml_config_file=self.yaml_config_file,
                                                          project_root=self.project_root,
                                                          fail_on_unknown_args=False)
        azure_config = AzureConfig(**parser_result.args)
        azure_config.project_root = self.project_root
        self.azure_config = azure_config
        self.model_config = None
        if not azure_config.model:
            raise ValueError("Parameter 'model' needs to be set to tell InnerEye which model to run.")
        model_config_loader: ModelConfigLoader = ModelConfigLoader(**parser_result.args)
        # Create the model as per the "model" commandline option. This can return either a built-in config
        # of type DeepLearningConfig, or a LightningContainer.
        config_or_container = model_config_loader.create_model_config_from_name(model_name=azure_config.model)

        def parse_overrides_and_apply(c: object, previous_parser_result: ParserResult) -> ParserResult:
            assert isinstance(c, GenericConfig)
            parser = type(c).create_argparser()
            # For each parser, feed in the unknown settings from the previous parser. All commandline args should
            # be consumed by name, hence fail if there is something that is still unknown.
            parser_result = parse_arguments(parser,
                                            settings_from_yaml=previous_parser_result.unknown_settings_from_yaml,
                                            args=previous_parser_result.unknown,
                                            fail_on_unknown_args=True)
            # Apply the overrides and validate. Overrides can come from either YAML settings or the commandline.
            c.apply_overrides(parser_result.known_settings_from_yaml)
            c.apply_overrides(parser_result.overrides)
            c.validate()
            return parser_result

        # Now create a parser that understands overrides at model/container level.
        parser_result = parse_overrides_and_apply(config_or_container, parser_result)

        if isinstance(config_or_container, LightningContainer):
            self.lightning_container = config_or_container
        elif isinstance(config_or_container, ModelConfigBase):
            # Built-in InnerEye models use a fake container
            self.model_config = config_or_container
            self.lightning_container = InnerEyeContainer(config_or_container)
        else:
            raise ValueError(f"Don't know how to handle a loaded configuration of type {type(config_or_container)}")

        # Allow overriding AzureConfig params from within the container.
        self.lightning_container.update_azure_config(self.azure_config)

        if azure_config.extra_code_directory:
            exist = "exists" if Path(azure_config.extra_code_directory).exists() else "does not exist"
            logging.info(f"extra_code_directory is {azure_config.extra_code_directory}, which {exist}")
        else:
            logging.info("extra_code_directory is unset")
        return parser_result

    def run(self) -> Tuple[Optional[DeepLearningConfig], AzureRunInfo]:
        """
        The main entry point for training and testing models from the commandline. This chooses a model to train
        via a commandline argument, runs training or testing, and writes all required info to disk and logs.
        :return: If submitting to AzureML, returns the model configuration that was used for training,
        including commandline overrides applied (if any).
        """
        # Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        logging_to_stdout(logging.INFO if is_local_rank_zero() else "ERROR")
        initialize_rpdb()
        user_agent.append(azure_util.INNEREYE_SDK_NAME, azure_util.INNEREYE_SDK_VERSION)
        self.parse_and_load_model()
        if self.lightning_container.perform_cross_validation:
            # force hyperdrive usage if performing cross validation
            self.azure_config.hyperdrive = True
        azure_run_info = self.submit_to_azureml_if_needed()
        self.run_in_situ(azure_run_info)
        if self.model_config is None:
            return self.lightning_container, azure_run_info
        return self.model_config, azure_run_info

    def submit_to_azureml_if_needed(self) -> AzureRunInfo:
        """
        Submit a job to AzureML, returning the resulting Run object, or exiting if we were asked to wait for
        completion and the Run did not succeed.
        """
        if self.azure_config.azureml and isinstance(self.model_config, DeepLearningConfig) \
                and not self.lightning_container.azure_dataset_id:
            raise ValueError("When running an InnerEye built-in model in AzureML, the 'azure_dataset_id' "
                             "property must be set.")
        source_config = SourceConfig(
            root_folder=self.project_root,
            entry_script=Path(sys.argv[0]).resolve(),
            script_params=sys.argv[1:],
            conda_dependencies_files=get_all_environment_files(self.project_root),
            hyperdrive_config_func=(self.model_config.get_hyperdrive_config if self.model_config
                                    else self.lightning_container.get_hyperdrive_config),
            # For large jobs, upload of results can time out because of large checkpoint files. Default is 600
            upload_timeout_seconds=86400
        )
        # Reduce the size of the snapshot by adding unused folders to amlignore. The Test* subfolders are only needed
        # when running pytest.
        ignored_folders = []
        if not self.azure_config.pytest_mark:
            ignored_folders.extend(["Tests", "TestsOutsidePackage"])
        if not self.lightning_container.regression_test_folder:
            ignored_folders.append("RegressionTestResults")

        all_local_datasets = self.lightning_container.all_local_dataset_paths()
        input_datasets = \
            create_dataset_configs(self.azure_config,
                                   all_azure_dataset_ids=self.lightning_container.all_azure_dataset_ids(),
                                   all_dataset_mountpoints=self.lightning_container.all_dataset_mountpoints(),
                                   all_local_datasets=all_local_datasets)  # type: ignore

        def after_submission_hook(azure_run: Run) -> None:
            """
            A function that will be called right after job submission.
            """
            # Add an extra tag that depends on the run that was actually submitted. This is used for later filtering
            # run in cross validation analysis
            recovery_id = create_run_recovery_id(azure_run)
            azure_run.tag(RUN_RECOVERY_ID_KEY_NAME, recovery_id)
            print("If this run fails, re-start runner.py and supply these additional arguments: "
                  f"--run_recovery_id={recovery_id}")
            if self.azure_config.tensorboard:
                print("Starting TensorBoard now because you specified --tensorboard")
                monitor(monitor_config=AMLTensorBoardMonitorConfig(run_ids=[azure_run.id]),
                        azure_config=self.azure_config)
            else:
                print(f"To monitor this run locally using TensorBoard, run the script: "
                      f"InnerEye/Azure/tensorboard_monitor.py --run_ids={azure_run.id}")

            if self.azure_config.wait_for_completion:
                # We want the job output to be visible on the console. Do not exit yet if the job fails, because we
                # may need to download the pytest result file.
                azure_run.wait_for_completion(show_output=True, raise_on_error=False)
                if self.azure_config.pytest_mark:
                    # The AzureML job can optionally run pytest. Attempt to download it to the current directory.
                    # A build step will pick up that file and publish it to Azure DevOps.
                    # If pytest_mark is set, this file must exist.
                    logging.info("Downloading pytest result file.")
                    download_pytest_result(azure_run)
                if azure_run.status == RunStatus.FAILED:
                    raise ValueError(f"The AzureML run failed. Please check this URL for details: "
                                     f"{azure_run.get_portal_url()}")

        hyperdrive_config = None
        if self.azure_config.hyperdrive:
            hyperdrive_config = self.lightning_container.get_hyperdrive_config(ScriptRunConfig(source_directory=""))

        # Create a temporary file for the merged conda file, that will be removed after submission of the job.
        temp_conda: Optional[Path] = None
        try:
            if len(source_config.conda_dependencies_files) > 1:
                temp_conda = source_config.root_folder / f"temp_environment-{uuid.uuid4().hex[:8]}.yml"
                # Merge the project-specific dependencies with the packages that InnerEye itself needs. This should not
                # be necessary if the innereye package is installed. It is necessary when working with an outer project
                # and InnerEye as a git submodule and submitting jobs from the local machine.
                # In case of version conflicts, the package version in the outer project is given priority.
                merge_conda_files(source_config.conda_dependencies_files, temp_conda)

            # Calls like `self.azure_config.get_workspace()` will fail if we have no AzureML credentials set up, and so
            # we should only attempt them if we intend to elevate this to AzureML
            if self.azure_config.azureml:
                if not self.azure_config.cluster:
                    raise ValueError("self.azure_config.cluster not set, but we need a compute_cluster_name to submit"
                                     "the script to run in AzureML")
                azure_run_info = submit_to_azure_if_needed(
                    entry_script=source_config.entry_script,
                    snapshot_root_directory=source_config.root_folder,
                    script_params=source_config.script_params,
                    conda_environment_file=temp_conda or source_config.conda_dependencies_files[0],
                    aml_workspace=self.azure_config.get_workspace(),
                    compute_cluster_name=self.azure_config.cluster,
                    environment_variables=source_config.environment_variables,
                    default_datastore=self.azure_config.azureml_datastore,
                    experiment_name=to_azure_friendly_string(create_experiment_name(self.azure_config)),
                    max_run_duration=self.azure_config.max_run_duration,
                    input_datasets=input_datasets,
                    num_nodes=self.azure_config.num_nodes,
                    wait_for_completion=False,
                    ignored_folders=ignored_folders,
                    pip_extra_index_url=self.azure_config.pip_extra_index_url,
                    submit_to_azureml=self.azure_config.azureml,
                    docker_base_image=DEFAULT_DOCKER_BASE_IMAGE,
                    docker_shm_size=self.azure_config.docker_shm_size,
                    tags=additional_run_tags(
                        azure_config=self.azure_config,
                        commandline_args=" ".join(source_config.script_params)),
                    after_submission=after_submission_hook,
                    hyperdrive_config=hyperdrive_config)
                # Set the default display name to what was provided as the "tag"
                if self.azure_config.tag:
                    azure_run_info.run.display_name = self.azure_config.tag
            else:
                # compute_cluster_name is a required parameter in early versions of the HI-ML package
                azure_run_info = submit_to_azure_if_needed(
                    input_datasets=input_datasets,
                    submit_to_azureml=False)
        finally:
            if temp_conda:
                temp_conda.unlink()
        # submit_to_azure_if_needed calls sys.exit after submitting to AzureML. We only reach this when running
        # the script locally or in AzureML.
        return azure_run_info

    def print_git_tags(self) -> None:
        """
        When running in AzureML, print all the tags that contain information about the git repository status,
        for answering the question "which code version was used" from a log file only.
        """
        git_tags = get_git_tags(self.azure_config)
        if is_offline_run_context(RUN_CONTEXT):
            # When running on a VM outside AzureML, we can read git information from the current repository
            tags_to_print = git_tags
        else:
            # When running in AzureML, the git repo information is not necessarily passed in, but we copy the git
            # information into run tags after submitting the job, and can read it out here.
            # Only print out those tags that were created from git-related information
            tags_to_print = {key: value for key, value in RUN_CONTEXT.get_tags().items() if key in git_tags}
        logging.info("Git repository information:")
        for key, value in tags_to_print.items():
            logging.info(f"    {key:20}: {value}")

    def run_in_situ(self, azure_run_info: AzureRunInfo) -> None:
        """
        Actually run the AzureML job; this method will typically run on an Azure VM.
        :param azure_run_info: Contains all information about the present run in AzureML, in particular where the
        datasets are mounted.
        """
        # Only set the logging level now. Usually, when we set logging to DEBUG, we want diagnostics about the model
        # build itself, but not the tons of debug information that AzureML submissions create.
        # Suppress the logging from all processes but the one for GPU 0 on each node, to make log files more readable
        logging_to_stdout(self.azure_config.log_level if is_local_rank_zero() else "ERROR")
        package_setup_and_hacks()
        if is_global_rank_zero():
            self.print_git_tags()
        # For the PR build in AzureML, we can either pytest, or the training of the simple PR model. Running both
        # only works when using DDP_spawn, but that has as a side-effect that it messes up memory consumption of the
        # large models.
        if self.azure_config.pytest_mark:
            outputs_folder = Path.cwd() / fixed_paths.DEFAULT_AML_UPLOAD_DIR
            pytest_passed, results_file_path = run_pytest(self.azure_config.pytest_mark, outputs_folder)
            if not pytest_passed:
                # Terminate if pytest has failed. This makes the smoke test in
                # PR builds fail if pytest fails.
                pytest_failures = f"Not all PyTest tests passed. See {results_file_path}"
                raise ValueError(pytest_failures)
        else:
            # Set environment variables for multi-node training if needed. This function will terminate early
            # if it detects that it is not in a multi-node environment.
            set_environment_variables_for_multi_node()
            self.ml_runner = self.create_ml_runner()
            self.ml_runner.setup(azure_run_info)
            self.ml_runner.run()

    def create_ml_runner(self) -> MLRunner:
        """
        Create and return an ML runner using the attributes of this Runner object.
        """
        return MLRunner(
            model_config=self.model_config,
            container=self.lightning_container,
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
        model_deployment_hook: Optional[ModelDeploymentHookSignature] = None) -> \
        Tuple[Optional[DeepLearningConfig], Optional[Run]]:
    """
    The main entry point for training and testing models from the commandline. This chooses a model to train
    via a commandline argument, runs training or testing, and writes all required info to disk and logs.
    :return: If submitting to AzureML, returns the model configuration that was used for training,
    including commandline overrides applied (if any). For details on the arguments, see the constructor of Runner.
    """
    runner = Runner(project_root, yaml_config_file, post_cross_validation_hook, model_deployment_hook)
    return runner.run()


def main() -> None:
    run(project_root=fixed_paths.repository_root_directory(),
        yaml_config_file=fixed_paths.SETTINGS_YAML_FILE,
        post_cross_validation_hook=default_post_cross_validation_hook)


if __name__ == '__main__':
    main()

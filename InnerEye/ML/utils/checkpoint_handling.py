#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
import os
import re
import tempfile
import time
import uuid
from builtins import property
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
from azureml.core import Model, Run, Workspace

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_util import RUN_CONTEXT, download_run_output_file, download_run_outputs_by_prefix, \
    fetch_child_runs, tag_values_all_distinct
from InnerEye.Common.common_util import OTHER_RUNS_SUBDIR_NAME
from InnerEye.Common.fixed_paths import DEFAULT_AML_UPLOAD_DIR, MODEL_INFERENCE_JSON_FILE_NAME
from InnerEye.ML.common import BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, CHECKPOINT_FOLDER, \
    LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX, RECOVERY_CHECKPOINT_FILE_NAME
from InnerEye.ML.deep_learning_config import OutputParams
from InnerEye.ML.lightning_container import LightningContainer
from InnerEye.ML.model_inference_config import read_model_inference_config
from InnerEye.ML.utils.run_recovery import RunRecovery
from health_azure import download_files_from_run_id, is_running_in_azure_ml
from health_azure.utils import get_run_file_names, is_global_rank_zero

MODEL_WEIGHTS_DIR_NAME = "trained_models"


class CheckpointHandler:
    """
    This class handles which checkpoints are used to initialize the model during train or test time based on the
    azure config and model config.
    """

    def __init__(self, container: LightningContainer, azure_config: AzureConfig,
                 project_root: Path, run_context: Optional[Run] = None):
        self.azure_config = azure_config
        self.container = container
        self.run_recovery: Optional[RunRecovery] = None
        self.project_root = project_root
        self.run_context = run_context
        self.trained_weights_paths: List[Path] = []
        self.has_continued_training = False

    @property
    def output_params(self) -> OutputParams:
        """
        Gets the part of the configuration that is responsible for output paths.
        """
        return self.container

    def download_checkpoints_from_hyperdrive_child_runs(self, hyperdrive_parent_run: Run) -> None:
        """
        Downloads the best checkpoints from all child runs of a Hyperdrive parent run. This is used to gather results
        for ensemble creation.
        """
        self.run_recovery = download_best_checkpoints_from_child_runs(self.output_params,
                                                                      hyperdrive_parent_run)
        # Check paths are good, just in case
        for path in self.run_recovery.checkpoints_roots:
            if not path.is_dir():
                raise NotADirectoryError(f"Does not exist or is not a directory: {path}")

    def download_recovery_checkpoints_or_weights(self, only_return_path: bool = False) -> None:
        """
        Download checkpoints from a run recovery object or from a weights url. Set the checkpoints path based on the
        run_recovery_object, weights_url or local_weights_path.
        This is called at the start of training.
        :param: only_return_path: if True, return a RunRecovery object with the path to the checkpoint without actually
        downloading the checkpoints. This is useful to avoid duplicating checkpoint download when running on multiple
        nodes. If False, return the RunRecovery object and download the checkpoint to disk.
        """
        if self.azure_config.run_recovery_id:
            run_to_recover = self.azure_config.fetch_run(self.azure_config.run_recovery_id.strip())
            self.run_recovery = download_all_checkpoints_from_run(self.output_params, run_to_recover,
                                                                  only_return_path=only_return_path)

        if self.container.weights_url or self.container.local_weights_path or self.container.model_id:
            self.trained_weights_paths = self.get_local_checkpoints_path_or_download()

    def additional_training_done(self) -> None:
        """
        Lets the handler know that training was done in this run.
        """
        self.has_continued_training = True

    def get_recovery_or_checkpoint_path_train(self) -> Optional[Path]:
        """
        Decides the checkpoint path to use for the current training run. Looks for the latest checkpoint in the
        checkpoint folder. If run_recovery is provided, the checkpoints will have been downloaded to this folder
        prior to calling this function. Else, if the run gets pre-empted and automatically restarted in AML,
        the latest checkpoint will be present in this folder too.
        :return: Constructed checkpoint path to recover from.
        """
        if is_global_rank_zero():
            checkpoints = list(self.container.checkpoint_folder.rglob("*"))
            logging.info(f"Available checkpoints: {len(checkpoints)}")
            for f in checkpoints:
                logging.info(f)
        recovery = find_recovery_checkpoint_and_epoch(self.container.checkpoint_folder)
        if recovery is not None:
            local_recovery_path, recovery_epoch = recovery
            self.container._start_epoch = recovery_epoch
            return local_recovery_path
        else:
            return None

    def get_best_checkpoints(self) -> List[Path]:
        """
        Get a list of checkpoints per epoch for testing/registration from the current training run.
        This function also checks that the checkpoint at the returned checkpoint path exists.
        """
        if not self.run_recovery and not self.has_continued_training:
            raise ValueError("Cannot recover checkpoint, no run recovery object provided and "
                             "no training has been done in this run.")

        checkpoint_paths = []
        if self.run_recovery:
            checkpoint_paths = self.run_recovery.get_best_checkpoint_paths()

            checkpoint_exists = []
            # Discard any checkpoint paths that do not exist - they will make inference/registration fail.
            # This can happen when some child runs in a hyperdrive run fail; it may still be worth running inference
            # or registering the model.
            for path in checkpoint_paths:
                if path.is_file():
                    checkpoint_exists.append(path)
                else:
                    logging.warning(f"Could not recover checkpoint path {path}")
            checkpoint_paths = checkpoint_exists

        if self.has_continued_training:
            # Checkpoint is from the current run, whether a new run or a run recovery which has been doing more
            # training, so we look for it there.
            checkpoint_from_current_run = self.output_params.get_path_to_best_checkpoint()
            if checkpoint_from_current_run.is_file():
                logging.info("Using checkpoints from current run.")
                checkpoint_paths = [checkpoint_from_current_run]
            else:
                logging.info("Training has continued, but not yet written a checkpoint. Using recovery checkpoints.")
        else:
            logging.info("Using checkpoints from run recovery")

        return checkpoint_paths

    def get_checkpoints_to_test(self) -> List[Path]:
        """
        Find the checkpoints to test. If a run recovery is provided, or if the model has been training, look for
        checkpoints corresponding to the epochs in get_test_epochs(). If there is no run recovery and the model was
        not trained in this run, then return the checkpoint from the local_weights_path.
        """

        checkpoints = []

        # If model was trained, look for the best checkpoint
        if self.run_recovery or self.has_continued_training:
            checkpoints = self.get_best_checkpoints()
        elif self.trained_weights_paths:
            # Model was not trained, check if there is a local weight path.
            logging.info(f"Using model weights from {self.trained_weights_paths} to initialize model")
            checkpoints = self.trained_weights_paths
        else:
            logging.warning("Could not find any local_weights_path, model_weights or model_id to get checkpoints from")

        return checkpoints

    @staticmethod
    def download_weights(urls: List[str], download_folder: Path) -> List[Path]:
        """
        Download a checkpoint from weights_url to the modelweights directory.
        """
        checkpoint_paths = []
        for url in urls:
            # assign the same filename as in the download url if possible, so that we can check for duplicates
            # If that fails, map to a random uuid
            file_name = os.path.basename(urlparse(url).path) or str(uuid.uuid4().hex)
            result_file = download_folder / file_name
            checkpoint_paths.append(result_file)
            # only download if hasn't already been downloaded
            if result_file.exists():
                logging.info(f"File already exists, skipping download: {result_file}")
            else:
                logging.info(f"Downloading weights from URL {url}")

                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(result_file, "wb") as file:
                    for chunk in response.iter_content(chunk_size=1024):
                        file.write(chunk)

        return checkpoint_paths

    @staticmethod
    def get_checkpoints_from_model(model_id: str, workspace: Workspace, download_path: Path) -> List[Path]:
        if len(model_id.split(":")) != 2:
            raise ValueError(
                f"model_id should be in the form 'model_name:version', got {model_id}")

        model_name, model_version = model_id.split(":")
        model = Model(workspace=workspace, name=model_name, version=int(model_version))
        model_path = Path(model.download(str(download_path), exist_ok=True))
        model_inference_config = read_model_inference_config(model_path / MODEL_INFERENCE_JSON_FILE_NAME)
        checkpoint_paths = [model_path / x for x in model_inference_config.checkpoint_paths]
        return checkpoint_paths

    def get_local_checkpoints_path_or_download(self) -> List[Path]:
        """
        Get the path to the local weights to use or download them and set local_weights_path
        """
        if not self.container.model_id and not self.container.local_weights_path and not self.container.weights_url:
            raise ValueError("Cannot download weights - none of model_id, local_weights_path or weights_url is set in "
                             "the model config.")

        if self.container.local_weights_path:
            checkpoint_paths = self.container.local_weights_path
        else:
            download_folder = self.output_params.checkpoint_folder / MODEL_WEIGHTS_DIR_NAME
            download_folder.mkdir(exist_ok=True, parents=True)

            if self.container.model_id:
                checkpoint_paths = CheckpointHandler.get_checkpoints_from_model(model_id=self.container.model_id,
                                                                                workspace=self.azure_config.get_workspace(),
                                                                                download_path=download_folder)
            elif self.container.weights_url:
                urls = self.container.weights_url
                checkpoint_paths = CheckpointHandler.download_weights(urls=urls,
                                                                      download_folder=download_folder)

        for checkpoint_path in checkpoint_paths:
            if not checkpoint_path or not checkpoint_path.is_file():
                raise FileNotFoundError(f"Could not find the weights file at {checkpoint_path}")
        return checkpoint_paths


def download_checkpoints_to_temp_folder(run: Optional[Run] = None, workspace: Optional[Workspace] = None) -> Path:
    """
    Downloads all files with the outputs/checkpoints prefix of the given run to a temporary folder.
    In distributed training, the download only happens once per node.

    :return: The path to which the files were downloaded.
    """
    run = run or RUN_CONTEXT
    # Downloads should go to a temporary folder because downloading the files to the checkpoint folder might
    # cause artifact conflicts later.
    temp_folder = Path(tempfile.mkdtemp())
    checkpoint_prefix = f"{DEFAULT_AML_UPLOAD_DIR}/{CHECKPOINT_FOLDER}/"
    existing_checkpoints = get_run_file_names(run, prefix=checkpoint_prefix)
    logging.info(f"Number of checkpoints available in AzureML: {len(existing_checkpoints)}")
    if len(existing_checkpoints) > 0:
        try:
            logging.info(f"Downloading checkpoints to {temp_folder}")
            download_files_from_run_id(run_id=run.id,
                                       output_folder=temp_folder,
                                       prefix=checkpoint_prefix,
                                       workspace=workspace)
        except Exception as ex:
            logging.warning(f"Unable to download checkpoints from AzureML. Error: {str(ex)}")
    return temp_folder


PathAndEpoch = Tuple[Path, int]


def find_recovery_checkpoint_and_epoch(path: Path) -> Optional[PathAndEpoch]:
    """
    Looks at all the recovery files, extracts the epoch number for all of them and returns the most recent (latest
    epoch)
    checkpoint path along with the corresponding epoch number. If no recovery checkpoint are found, return None.
    :param path: The folder to start searching in.
    :return: None if there is no file matching the search pattern, or a Tuple with Path object and integer pointing to
    recovery checkpoint path and recovery epoch.
    """
    available_checkpoints = find_all_recovery_checkpoints(path)
    if available_checkpoints is None and is_running_in_azure_ml():
        logging.info("No recovery checkpoints available in the checkpoint folder. Trying to find checkpoints in "
                     "AzureML from previous runs of this job.")
        # Download checkpoints from AzureML, then try to find recovery checkpoints among those.
        temp_folder = download_checkpoints_to_temp_folder()
        available_checkpoints = find_all_recovery_checkpoints(temp_folder)
    if available_checkpoints is not None:
        return extract_latest_checkpoint_and_epoch(available_checkpoints)
    return None


def get_recovery_checkpoint_path(path: Path) -> Path:
    """
    Returns the path to the last recovery checkpoint in the given folder or the provided filename. Raises a
    FileNotFoundError if no
    recovery checkpoint file is present.
    :param path: Path to checkpoint folder
    """
    recovery_ckpt_and_epoch = find_recovery_checkpoint_and_epoch(path)
    if recovery_ckpt_and_epoch is not None:
        return recovery_ckpt_and_epoch[0]
    files = list(path.glob("*"))
    raise FileNotFoundError(f"No checkpoint files found in {path}. Existing files: {' '.join(p.name for p in files)}")


def find_all_recovery_checkpoints(path: Path) -> Optional[List[Path]]:
    """
    Extracts all file starting with RECOVERY_CHECKPOINT_FILE_NAME in path
    :param path:
    :return:
    """
    all_recovery_files = [f for f in path.glob(RECOVERY_CHECKPOINT_FILE_NAME + "*")]
    if len(all_recovery_files) == 0:
        return None
    return all_recovery_files


def extract_latest_checkpoint_and_epoch(available_files: List[Path]) -> PathAndEpoch:
    """
     Checkpoints are saved as recovery_epoch={epoch}.ckpt, find the latest ckpt and epoch number.
    :param available_files: all available checkpoints
    :return: path the checkpoint from latest epoch and epoch number
    """
    recovery_epochs = [int(re.findall(r"[\d]+", f.stem)[0]) for f in available_files]
    idx_max_epoch = int(np.argmax(recovery_epochs))
    return available_files[idx_max_epoch], recovery_epochs[idx_max_epoch]


def create_best_checkpoint(path: Path) -> Path:
    """
    Creates the best checkpoint file. "Best" is at the moment defined as being the last checkpoint, but could be
    based on some defined policy.
    The best checkpoint will be renamed to `best_checkpoint.ckpt`.
    :param path: The folder that contains all checkpoint files.
    """
    logging.debug(f"Files in checkpoint folder: {' '.join(p.name for p in path.glob('*'))}")
    last_ckpt = path / LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    all_files = f"Existing files: {' '.join(p.name for p in path.glob('*'))}"
    if not last_ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint file {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} not found. {all_files}")
    logging.info(f"Using {LAST_CHECKPOINT_FILE_NAME_WITH_SUFFIX} as the best checkpoint: Renaming to "
                 f"{BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX}")
    best = path / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX
    last_ckpt.rename(best)
    return best


def download_best_checkpoints_from_child_runs(config: OutputParams, run: Run) -> RunRecovery:
    """
    Downloads the best checkpoints from all child runs of the provided Hyperdrive parent run.
    The checkpoints for the sibling runs will go into folder 'OTHER_RUNS/<cross_validation_split>'
    in the checkpoint folder. There is special treatment for the child run that is equal to the present AzureML
    run, its checkpoints will be read off the checkpoint folder as-is.
    :param config: Model related configs.
    :param run: The Hyperdrive parent run to download from.
    :return: run recovery information
    """
    child_runs: List[Run] = fetch_child_runs(run)
    if not child_runs:
        raise ValueError(f"AzureML run {run.id} does not have any child runs.")
    logging.info(f"Run {run.id} has {len(child_runs)} child runs: {', '.join(c.id for c in child_runs)}")
    tag_to_use = 'cross_validation_split_index'
    can_use_split_indices = tag_values_all_distinct(child_runs, tag_to_use)
    # download checkpoints for the child runs in the root of the parent
    child_runs_checkpoints_roots: List[Path] = []
    for child in child_runs:
        if child.id == RUN_CONTEXT.id:
            # We expect to find the file(s) we need in config.checkpoint_folder
            child_dst = config.checkpoint_folder
        else:
            subdir = str(child.tags[tag_to_use] if can_use_split_indices else child.number)
            child_dst = config.checkpoint_folder / OTHER_RUNS_SUBDIR_NAME / subdir
            download_run_output_file(
                blob_path=Path(CHECKPOINT_FOLDER) / BEST_CHECKPOINT_FILE_NAME_WITH_SUFFIX,
                destination=child_dst,
                run=child
            )
        child_runs_checkpoints_roots.append(child_dst)
    return RunRecovery(checkpoints_roots=child_runs_checkpoints_roots)


def download_all_checkpoints_from_run(config: OutputParams, run: Run,
                                      subfolder: Optional[str] = None,
                                      only_return_path: bool = False) -> RunRecovery:
    """
    Downloads all checkpoints of the provided run inside the checkpoints folder.
    :param config: Model related configs.
    :param run: Run whose checkpoints should be recovered
    :param subfolder: optional subfolder name, if provided the checkpoints will be downloaded to
    CHECKPOINT_FOLDER / subfolder. If None, the checkpoint are downloaded to CHECKPOINT_FOLDER of the current run.
    :param: only_return_path: if True, return a RunRecovery object with the path to the checkpoint without actually
    downloading the checkpoints. This is useful to avoid duplicating checkpoint download when running on multiple
    nodes. If False, return the RunRecovery object and download the checkpoint to disk.
    :return: run recovery information
    """
    if fetch_child_runs(run):
        raise ValueError(f"AzureML run {run.id} has child runs, this method does not support those.")

    destination_folder = config.checkpoint_folder / subfolder if subfolder else config.checkpoint_folder

    if not only_return_path:
        download_run_outputs_by_prefix(
            blobs_prefix=Path(CHECKPOINT_FOLDER),
            destination=destination_folder,
            run=run
        )
    time.sleep(60)  # Needed because AML is not fast enough to download
    return RunRecovery(checkpoints_roots=[destination_folder])

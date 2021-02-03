#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import param
import requests
from azureml.core import Experiment, Model

from InnerEye.Azure.azure_config import AzureConfig, SourceConfig
from InnerEye.Azure.azure_runner import create_run_config
from InnerEye.Common import fixed_paths
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME, ENVIRONMENT_YAML_FILE_NAME
from InnerEye.Common.generic_parsing import GenericConfig


class SubmitForInferenceConfig(GenericConfig):
    """
    Command line parameter class.
    """
    experiment_name: str = param.String(default="model_inference",
                                        doc="Name of experiment the run should belong to")
    model_id: str = param.String(doc="Id of model, e.g. Prostate:123. Mandatory.")
    image_file: Path = param.ClassSelector(class_=Path, doc="Image file to segment, ending in .nii.gz. Mandatory.")
    settings: Path = param.ClassSelector(class_=Path,
                                         doc="File containing Azure settings (typically your settings.yml). If not "
                                             "provided, use the default settings file.")
    download_folder: Optional[Path] = \
        param.ClassSelector(class_=Path, default=None,
                            doc="Folder into which to download the segmentation result. If this is provided, the "
                                "script waits for the AzureML run to complete.")
    keep_upload_folder: bool = param.Boolean(
        default=False, doc="Whether to keep the temporary upload folder after the inference run is submitted")
    cluster: str = param.String(doc="The name of the GPU cluster in which to run the experiment. If not provided, use "
                                    "the cluster in the settings.yml file")

    def validate(self) -> None:
        # The image file must be specified, must exist, and must end in .nii.gz, i.e. be
        # a compressed Nifti file.
        assert self.image_file
        if not self.image_file.is_file():
            raise FileNotFoundError(self.image_file)
        basename = str(self.image_file.name)
        # Do not import the pre-defined constants from io_util here, so that we can keep the Conda environment for
        # running tests_after_training.py small.
        extension = ".nii.gz"
        if not basename.endswith(extension):
            raise ValueError(f"Bad image file name, does not end with {extension}: {self.image_file.name}")
        # If the user wants the result downloaded, the download folder must already exist
        if self.download_folder is not None:
            assert self.download_folder.exists()


def copy_image_file(image: Path, destination_folder: Path) -> Path:
    """
    Copy the source image file into the given folder destination_folder.
    :param image: image file, must be Gzipped Nifti format with name ending .nii.gz
    :param destination_folder: top-level directory to copy image into (as test.nii.gz)
    :return: The full path of the image in the destination_folder
    """
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / fixed_paths.DEFAULT_TEST_IMAGE_NAME
    logging.info(f"Copying {image} to {destination}")
    shutil.copyfile(str(image), str(destination))
    return destination


def download_files_from_model(model_sas_urls: Dict[str, str], base_name: str, dir_path: Path) -> List[Path]:
    """
    Identifies all the files in an AzureML model that have a given file name (ignoring path), and downloads them
    to a folder.
    :param model_sas_urls: The files making up the model, as a mapping from file name to a URL with
    an SAS token.
    :param base_name: The file name of the files to download.
    :param dir_path: The folder into which the files will be written. All downloaded files will keep the relative
    path that they also have in the model.
    :return: a list of the files that were downloaded.
    """
    downloaded: List[Path] = []
    for path, url in model_sas_urls.items():
        if Path(path).name == base_name:
            target_path = dir_path / path
            target_path.parent.mkdir(exist_ok=True, parents=True)
            target_path.write_bytes(requests.get(url, allow_redirects=True).content)
            # Remove additional information from the URL to make it more legible
            index = url.find("?")
            if index > 0:
                url = url[:index]
            logging.info(f"Downloaded {path} from {url}")
            downloaded.append(target_path)
    if not downloaded:
        logging.warning(f"No file(s) with name '{base_name}' were found in the model!")
    return downloaded


def choose_download_path(download_folder: Path) -> Path:
    """
    Returns the path of a file in download_folder that does already exist. The first path tried is
    download_folder/segmentation.nii.gz, but if that does not exist, the names segmentation_001.nii.gz,
    segmentation_002.nii.gz etc are tried.
    """
    index = 0
    base = DEFAULT_RESULT_IMAGE_NAME
    while True:
        path = download_folder / base
        if not path.exists():
            return path
        index += 1
        components = DEFAULT_RESULT_IMAGE_NAME.split(".")
        base = ".".join([f"{components[0]}_{index:03d}"] + components[1:])


def submit_for_inference(args: SubmitForInferenceConfig, azure_config: AzureConfig) -> Optional[Path]:
    """
    Create and submit an inference to AzureML, and optionally download the resulting segmentation.
    :param azure_config: An object with all necessary information for accessing Azure.
    :param args: configuration, see SubmitForInferenceConfig
    :return: path to downloaded segmentation on local disc, or None if none.
    """
    logging.info(f"Building Azure configuration from {args.settings}")
    logging.info("Getting workspace")
    workspace = azure_config.get_workspace()
    logging.info("Identifying model")
    model = Model(workspace=workspace, id=args.model_id)
    model_id = model.id
    logging.info(f"Identified model {model_id}")
    source_directory = tempfile.TemporaryDirectory()
    source_directory_path = Path(source_directory.name)
    logging.info(f"Building inference run submission in {source_directory_path}")
    image_folder = source_directory_path / fixed_paths.DEFAULT_DATA_FOLDER
    image = copy_image_file(args.image_file, image_folder)
    model_sas_urls = model.get_sas_urls()
    # Identifies all the files with basename "environment.yml" in the model and downloads them.
    # These downloads should go into a temp folder that will most likely not be included in the model itself,
    # because the AzureML run will later download the model into the same folder structure, and the file names might
    # clash.
    temp_folder = source_directory_path / "temp_for_scoring"
    conda_files = download_files_from_model(model_sas_urls, ENVIRONMENT_YAML_FILE_NAME, dir_path=temp_folder)
    if not conda_files:
        raise ValueError("At least 1 Conda environment definition must exist in the model.")
    # Copy the scoring script from the repository. This will start the model download from Azure, and invoke the
    # scoring script.
    entry_script = source_directory_path / Path(fixed_paths.RUN_SCORING_SCRIPT).name
    shutil.copyfile(str(fixed_paths.repository_root_directory(fixed_paths.RUN_SCORING_SCRIPT)),
                    str(entry_script))
    source_config = SourceConfig(
        root_folder=source_directory_path,
        entry_script=entry_script,
        script_params=["--model-folder", ".",
                       "--model-id", model_id,
                       fixed_paths.SCORE_SCRIPT,
                       # The data folder must be relative to the root folder of the AzureML job. test_image_files
                       # is then just the file relative to the data_folder
                       "--data_folder", image.parent.name,
                       "--image_files", image.name],
        conda_dependencies_files=conda_files,
    )
    run_config = create_run_config(azure_config, source_config)
    exp = Experiment(workspace=workspace, name=args.experiment_name)
    run = exp.submit(run_config)
    logging.info(f"Submitted run {run.id} in experiment {run.experiment.name}")
    logging.info(f"Run URL: {run.get_portal_url()}")
    if not args.keep_upload_folder:
        source_directory.cleanup()
        logging.info(f"Deleted submission directory {source_directory_path}")
    if args.download_folder is None:
        return None
    logging.info("Awaiting run completion")
    run.wait_for_completion()
    logging.info(f"Run has completed with status {run.get_status()}")
    download_path = choose_download_path(args.download_folder)
    logging.info(f"Attempting to download segmentation to {download_path}")
    run.download_file(DEFAULT_RESULT_IMAGE_NAME, str(download_path))
    if download_path.exists():
        logging.info(f"Downloaded segmentation to {download_path}")
    else:
        logging.warning("Segmentation NOT downloaded")
    return download_path


def main(args: Optional[List[str]] = None, project_root: Optional[Path] = None) -> None:
    """
    Main function.
    """
    logging_to_stdout()
    inference_config = SubmitForInferenceConfig.parse_args(args)
    settings = inference_config.settings or fixed_paths.SETTINGS_YAML_FILE
    azure_config = AzureConfig.from_yaml(settings, project_root=project_root)
    if inference_config.cluster:
        azure_config.cluster = inference_config.cluster
    submit_for_inference(inference_config, azure_config)


if __name__ == '__main__':
    main(project_root=fixed_paths.repository_root_directory())

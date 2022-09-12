#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

innereye_root = Path(__file__).resolve().parent.parent.parent
if (innereye_root / "InnerEye").is_dir():
    innereye_root_str = str(innereye_root)
    if innereye_root_str not in sys.path:
        logging.info("Adding InnerEye folder to sys.path: %s", innereye_root_str)
        sys.path.insert(0, innereye_root_str)

import param
import requests
from azureml.core import Model, ScriptRunConfig
from health_azure import create_run_configuration, submit_run

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.fixed_paths import (
    DEFAULT_DATA_FOLDER, DEFAULT_RESULT_IMAGE_NAME, DEFAULT_RESULT_ZIP_DICOM_NAME,
    DEFAULT_TEST_IMAGE_NAME, DEFAULT_TEST_ZIP_NAME, PYTHON_ENVIRONMENT_NAME,
    RUN_SCORING_SCRIPT, SCORE_SCRIPT, SETTINGS_YAML_FILE, repository_root_directory
)
from InnerEye.Common.generic_parsing import GenericConfig


class SubmitForInferenceConfig(GenericConfig):
    """
    Command line parameter class.
    """
    experiment_name: str = param.String(default="model_inference",
                                        doc="Name of experiment the run should belong to")
    model_id: str = param.String(doc="Id of model, e.g. Prostate:123. Mandatory.")
    image_file: Path = param.ClassSelector(class_=Path,
                                           doc="Image file to segment, ending in .nii.gz if use_dicom=False, "
                                               "or zip of a DICOM series otherwise. Mandatory.")
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
    use_dicom: bool = param.Boolean(False, doc="If image to be segmented is a DICOM series and output to be DICOM-RT. "
                                               "If this is set then image_file should be a zip file "
                                               "containing a set of DICOM files.")

    def validate(self) -> None:
        # The image file must be specified, must exist, and must end in .nii.gz, i.e. be
        # a compressed Nifti file if use_dicom=False, or end in .zip otherwise.
        assert self.image_file
        if not self.image_file.is_file():
            raise FileNotFoundError(self.image_file)
        basename = str(self.image_file.name)
        # Do not import the pre-defined constants from io_util here, so that we can keep the Conda environment for
        # running tests_after_training.py small.
        extension = ".zip" if self.use_dicom else ".nii.gz"
        if not basename.endswith(extension):
            raise ValueError(f"Bad image file name, does not end with {extension}: {self.image_file.name}")
        # If the user wants the result downloaded, the download folder must already exist
        if self.download_folder is not None:
            assert self.download_folder.exists()


def copy_image_file(image: Path, destination_folder: Path, use_dicom: bool) -> Path:
    """
    Copy the source image file into the given folder destination_folder.

    :param image: image file, must be Gzipped Nifti format with name ending .nii.gz if use_dicom=False or .zip
        otherwise.

    :param destination_folder: top-level directory to copy image into (as test.nii.gz or test.zip)
    :param use_dicom: True to treat as a zip file.
    :return: The full path of the image in the destination_folder
    """
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = destination_folder / (DEFAULT_TEST_ZIP_NAME if use_dicom else DEFAULT_TEST_IMAGE_NAME)
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


def choose_download_path(result_image_name: str, download_folder: Path) -> Path:
    """
    Find a path to a file similiar to result_image_name that does not already exist in download_folder.

    The first path tried is download_folder/result_image_name, but if that exists, try appending
    _001, _002, etc to the file name until a new filename is found.

    :param result_image_name: Target filename.
    :param download_folder: Target folder.
    :return: Path to a file in download_folder that does not exist.
    """
    index = 0
    base = result_image_name
    while True:
        path = download_folder / base
        if not path.exists():
            return path
        index += 1
        components = result_image_name.split(".")
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
    image_folder = source_directory_path / DEFAULT_DATA_FOLDER
    image = copy_image_file(args.image_file, image_folder, args.use_dicom)
    # Retrieve the name of the Python environment that the training run used. This environment should have been
    # registered at training.
    python_environment_name = model.tags.get(PYTHON_ENVIRONMENT_NAME, "")
    if not python_environment_name:
        raise ValueError(f"The model did not contain tag {PYTHON_ENVIRONMENT_NAME} for the AzureML environment to use.")
    # Copy the scoring script from the repository. This will start the model download from Azure, and invoke the
    # scoring script.
    entry_script = source_directory_path / Path(RUN_SCORING_SCRIPT).name
    shutil.copyfile(str(repository_root_directory(RUN_SCORING_SCRIPT)),
                    str(entry_script))
    run_config = create_run_configuration(workspace=azure_config.get_workspace(),
                                          compute_cluster_name=azure_config.cluster,
                                          aml_environment_name=python_environment_name)
    script_run_config = ScriptRunConfig(
        source_directory=str(source_directory_path),
        script=entry_script.relative_to(source_directory_path),
        arguments=["--model-folder", ".",
                   "--model-id", model_id,
                   SCORE_SCRIPT,
                   # The data folder must be relative to the root folder of the AzureML
                   # job. image_files is then just the file relative to the data_folder
                   "--data_folder", image.parent.name,
                   "--image_files", image.name,
                   "--use_dicom", str(args.use_dicom),
                   "--model_id", model_id],
        run_config=run_config
    )

    run = submit_run(workspace=workspace,
                     experiment_name=args.experiment_name,
                     script_run_config=script_run_config,
                     wait_for_completion=True)
    if not args.keep_upload_folder:
        source_directory.cleanup()
        logging.info(f"Deleted submission directory {source_directory_path}")
    if args.download_folder is None:
        return None
    logging.info(f"Run has completed with status {run.get_status()}")
    download_file = DEFAULT_RESULT_ZIP_DICOM_NAME if args.use_dicom else DEFAULT_RESULT_IMAGE_NAME
    download_path = choose_download_path(download_file, args.download_folder)
    logging.info(f"Attempting to download segmentation to {download_path}")
    run.download_file(download_file, str(download_path))
    if download_path.exists():
        logging.info(f"Downloaded segmentation to {download_path}")
    else:
        logging.warning("Segmentation NOT downloaded")
    return download_path


def get_submit_for_inference_parser() -> argparse.ArgumentParser:
    """This function is need to allow sphinx to access the arguments for documenation

    :return: An example parser used for inference submission
    """

    inference_config = SubmitForInferenceConfig(should_validate=False)
    return inference_config.create_argparser()


def main(args: Optional[List[str]] = None, project_root: Optional[Path] = None) -> None:
    """
    Main function.
    """
    logging_to_stdout()
    inference_config = SubmitForInferenceConfig.parse_args(args)
    settings = inference_config.settings or SETTINGS_YAML_FILE
    azure_config = AzureConfig.from_yaml(settings, project_root=project_root)
    if inference_config.cluster:
        azure_config.cluster = inference_config.cluster
    submit_for_inference(inference_config, azure_config)


if __name__ == '__main__':
    main(project_root=repository_root_directory())

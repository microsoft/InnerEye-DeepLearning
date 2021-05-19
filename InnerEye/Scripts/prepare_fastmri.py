#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Union

from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import (ActivityDependency,
                                           AzureBlobStorageLinkedService,
                                           AzureBlobStorageLocation,
                                           BinaryDataset,
                                           BinaryReadSettings,
                                           BinarySource,
                                           BlobSink,
                                           CopyActivity,
                                           DatasetReference,
                                           DatasetResource,
                                           DatasetTarCompression,
                                           DatasetTarGZipCompression,
                                           Factory,
                                           HttpLinkedService,
                                           HttpServerLocation,
                                           HttpSource,
                                           LinkedServiceReference,
                                           LinkedServiceResource,
                                           PipelineResource,
                                           RunFilterParameters,
                                           SecureString,
                                           TarGZipReadSettings,
                                           TarReadSettings)

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.secrets_handling import SecretsHandling
from InnerEye.Common import fixed_paths

# The Azure blob container that will hold all downloaded results
TARGET_CONTAINER = "datasets"
# Datasets get downloaded in compressed and decompressed format. The compresed datasets have the same name as the
# decompressed folder (for example, 'knee_singlecoil'), but with this suffix ('knee_singlecoil_compressed')
COMPRESSED_DATASET_SUFFIX = "_compressed"

# Mapping different of the raw tar.gz files to separate dataset folders.
# First tuple item is the target folder in blob storage, then comes a list of raw files to extract into that folder.
# If an any in the list of raw files is a tuple, then the file extension is misleading:
# Some of the files that appear to be .tar.gz are actually plain .tar file.
FolderAndFileList = List[Tuple[str, List[Union[str, Tuple[str, str]]]]]
files_to_download: FolderAndFileList = [
    ("knee_singlecoil", [("knee_singlecoil_train.tar.gz", ".tar"),
                         ("knee_singlecoil_val.tar.gz", ".tar"),
                         ("knee_singlecoil_test_v2.tar.gz", ".tar"),
                         "knee_singlecoil_challenge.tar.gz"]),
    ("knee_multicoil", [("multicoil_train.tar.gz", ".tar"),
                        ("multicoil_val.tar.gz", ".tar"),
                        ("knee_multicoil_test_v2.tar.gz", ".tar"),
                        "knee_multicoil_challenge.tar.gz"]),
    ("knee_DICOMs", [
        "knee_mri_dicom_batch1.tar",
        "knee_mri_dicom_batch2.tar"
    ]),
    ("brain_multicoil", ["brain_multicoil_train.tar.gz",
                         "brain_multicoil_val.tar.gz",
                         "brain_multicoil_test.tar.gz",
                         "brain_multicoil_challenge.tar.gz",
                         "brain_multicoil_challenge_transfer.tar.gz"]),
    ("brain_DICOMs", [
        "brain_fastMRI_DICOM.tar.gz"
    ])
]


def get_azure_auth(azure_config: AzureConfig) -> Union[DefaultAzureCredential, ClientSecretCredential]:
    """
    Returns the authentication object for the azure.identity library,
    based on either the chosen Service Principal (if set, and if the password was found), or the
    interactive browser authentication if not all Service Principal information is available.
    :param azure_config: The object containing all Azure-related information.
    :return: An azure.identity authentication object.
    """
    secrets_handler = SecretsHandling(project_root=azure_config.project_root)
    application_key = secrets_handler.get_secret_from_environment(fixed_paths.SERVICE_PRINCIPAL_KEY,
                                                                  allow_missing=True)
    if not azure_config.tenant_id:
        raise ValueError("No tenant_id field was found. Please complete the Azure setup.")
    if application_key and azure_config.application_id:
        return ClientSecretCredential(
            tenant_id=azure_config.tenant_id,
            client_id=azure_config.application_id,
            client_secret=application_key)

    logging.warning("Unable to retrieve the key for the Service Principal authentication "
                    f"(expected in environment variable '{fixed_paths.SERVICE_PRINCIPAL_KEY}' or YAML). "
                    f"Switching to interactive login.")
    return DefaultAzureCredential()


def create_datafactory_and_run(files_and_tokens: Dict[str, str],
                               connection_string: str,
                               location: str,
                               is_unittest: bool = False) -> None:
    """
    Builds an Azure Data Factory to download the FastMRI dataset from AWS, and places them in Azure Blob Storage.
    :param location: The Azure location in which the Data Factory should be created (for example, "westeurope")
    :param files_and_tokens: A mapping from file name (like knee.tar.gz) to AWS access token.
    :param is_unittest: If True, download a small tar.gz file from github. If False, download the "real" fastMRI
    datafiles from AWS.
    :param connection_string: The connection string of the Azure storage where the downloaded data should be stored.
    """

    azure_config = AzureConfig.from_yaml(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())

    # The data factory name. It must be globally unique.
    data_factory_name = "fastmri-copy-data-" + uuid.uuid4().hex[:8]

    # Get either the Service Principal authentication, if those are set already, or use interactive auth in the browser
    azureid_auth = get_azure_auth(azure_config)

    # Create a data factory
    adf_client = DataFactoryManagementClient(azureid_auth, azure_config.subscription_id)
    df_resource = Factory(location=location)
    print(f"Creating data factory {data_factory_name}")
    df = adf_client.factories.create_or_update(azure_config.resource_group, data_factory_name, df_resource)
    while df.provisioning_state != 'Succeeded':
        df = adf_client.factories.get(azure_config.resource_group, data_factory_name)
        time.sleep(1)
    print("Data factory created")

    # Create a linked service pointing to where the downloads come from
    if is_unittest:
        http_service = LinkedServiceResource(
            properties=HttpLinkedService(url="https://github.com",
                                         enable_server_certificate_validation=True,
                                         authentication_type="Anonymous"))
    else:
        http_service = LinkedServiceResource(
            properties=HttpLinkedService(url="https://fastmri-dataset.s3.amazonaws.com/",
                                         enable_server_certificate_validation=True,
                                         authentication_type="Anonymous"))
    http_name = "AwsHttp"
    adf_client.linked_services.create_or_update(resource_group_name=azure_config.resource_group,
                                                factory_name=data_factory_name,
                                                linked_service_name=http_name,
                                                linked_service=http_service)
    # Create a linked service that represents the sink (Azure blob storage)
    blob_storage_name = "AzureBlob"
    blob_storage = AzureBlobStorageLinkedService(connection_string=SecureString(value=connection_string))
    blob_storage_service = LinkedServiceResource(properties=blob_storage)
    adf_client.linked_services.create_or_update(resource_group_name=azure_config.resource_group,
                                                factory_name=data_factory_name,
                                                linked_service_name=blob_storage_name,
                                                linked_service=blob_storage_service)

    linked_blob_storage = LinkedServiceReference(reference_name=blob_storage_name)
    linked_http = LinkedServiceReference(reference_name=http_name)

    def download_and_uncompress(source_file_or_tuple: Union[str, Tuple[str, str]], target_folder: str) -> List[str]:
        """
        Downloads a file from AWS and stores them in blob storage in its compressed form.
        From the compressed file in blob storage, it is then uncompressed, and written to a new folder in blob storage.
        For example, if 'target_folder' is 'foo', the uncompressed file will be written to folder 'foo', and the
        compressed raw data will be written to 'foo_compressed'.
        :param source_file_or_tuple: The name of the .tar.gz or .tar file to download, without any access tokens.
        If the name is a Tuple[str, str], the second tuple element is the "real" extension, for files where the
        extension is misleading.
        :param target_folder: The folder prefix in the target storage account.
        :return: A list of pipelines that this method created.
        """
        if isinstance(source_file_or_tuple, str):
            source_file = source_file_or_tuple
            file_extension = "".join(Path(source_file).suffixes)
            correct_extension = file_extension
        elif isinstance(source_file_or_tuple, tuple):
            source_file, correct_extension = source_file_or_tuple
            file_extension = "".join(Path(source_file).suffixes)
        else:
            raise ValueError(f"Type of source_file_or_tuple not recognized: {type(source_file_or_tuple)}")
        source_file_with_correct_extension = source_file[:source_file.rfind(file_extension)] + correct_extension
        target_folder_compressed = target_folder + COMPRESSED_DATASET_SUFFIX
        if is_unittest:
            http_source = HttpServerLocation(relative_url="gulpjs/gulp/archive/v3.9.1.tar.gz")
        else:
            http_source = HttpServerLocation(relative_url=f"{source_file}{files_and_tokens[source_file]}")
        source_file_cleaned = source_file.replace(".", "_")
        # A dataset that reads the files from AWS as-is, no decompression
        source_compressed = BinaryDataset(linked_service_name=linked_http,
                                          location=http_source)
        source_compressed_name = f"{source_file_cleaned} on AWS"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=source_compressed_name,
                                             dataset=DatasetResource(properties=source_compressed))
        # The sink for downloading the datasets as-is (compressed)
        blob_storage_compressed = AzureBlobStorageLocation(file_name=source_file_with_correct_extension,
                                                           container=TARGET_CONTAINER,
                                                           folder_path=target_folder_compressed)
        dest_compressed = BinaryDataset(linked_service_name=linked_blob_storage,
                                        location=blob_storage_compressed)
        dest_compressed_name = f"{source_file_cleaned} on Azure"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=dest_compressed_name,
                                             dataset=DatasetResource(properties=dest_compressed))
        # A dataset that reads the files from blob storage and uncompresses on-the-fly
        if correct_extension == ".tar.gz":
            compression = DatasetTarGZipCompression()
            # By default, a folder gets created for each .tar.gzip file that is read. Disable that.
            compression_properties = TarGZipReadSettings(preserve_compression_file_name_as_folder=False)
        elif correct_extension == ".tar":
            compression = DatasetTarCompression()
            # By default, a folder gets created for each .tar file that is read. Disable that.
            compression_properties = TarReadSettings(preserve_compression_file_name_as_folder=False)
        else:
            raise ValueError(f"Unable to determine compression for file {source_file}")
        source_uncompressed = BinaryDataset(linked_service_name=linked_blob_storage,
                                            location=blob_storage_compressed,
                                            compression=compression)
        source_uncompressed_name = f"read {source_file_cleaned} and uncompress"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=source_uncompressed_name,
                                             dataset=DatasetResource(properties=source_uncompressed))
        # The sink for downloading the datasets uncompressed
        final_dataset = BinaryDataset(linked_service_name=linked_blob_storage,
                                      location=AzureBlobStorageLocation(container=TARGET_CONTAINER,
                                                                        folder_path=target_folder))
        final_name = f"save {source_file_cleaned} uncompressed"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=final_name,
                                             dataset=DatasetResource(properties=final_dataset))
        # Copying from compressed source to compressed destination on blob storage
        download = CopyActivity(name=f"download {source_file_cleaned}",
                                inputs=[DatasetReference(reference_name=source_compressed_name)],
                                outputs=[DatasetReference(reference_name=dest_compressed_name)],
                                source=HttpSource(),
                                sink=BlobSink())
        # Read the compressed file from blob storage and create an uncompressed dataset.
        # This should not create extra folder structure beyond what is already in the tar file - this is specified
        # in compression_properties
        binary_source = BinarySource(format_settings=BinaryReadSettings(compression_properties=compression_properties))
        uncompress = CopyActivity(name=f"uncompress {source_file_cleaned}",
                                  inputs=[DatasetReference(reference_name=source_uncompressed_name)],
                                  outputs=[DatasetReference(reference_name=final_name)],
                                  source=binary_source,
                                  sink=BlobSink(),
                                  # Add a dependent activity: We first need to download
                                  depends_on=[
                                      ActivityDependency(activity=download.name, dependency_conditions=["Succeeded"])]
                                  )
        # Create a pipeline that first downloads from AWS to blob storage, and then decompresses from blob storage
        # to another blob storage location
        pipeline = f"{source_file_cleaned} to folder {target_folder}"
        adf_client.pipelines.create_or_update(resource_group_name=azure_config.resource_group,
                                              factory_name=data_factory_name,
                                              pipeline_name=pipeline,
                                              pipeline=PipelineResource(activities=[download, uncompress]))
        return [pipeline]

    file_list: FolderAndFileList = \
        [("antonsctest", ["foo.tar.gz", "bar.tar"])] if is_unittest else files_to_download
    all_pipelines = []
    print("Creating pipelines:")
    for target_folder, files in file_list:
        for file in files:
            pipelines = download_and_uncompress(file, target_folder=target_folder)
            for p in pipelines:
                print(f"Created pipeline {p}")
            all_pipelines.extend(pipelines)

    print("Starting all pipelines")
    run_ids_per_pipeline = {}
    for pipeline in all_pipelines:
        run_result = adf_client.pipelines.create_run(resource_group_name=azure_config.resource_group,
                                                     factory_name=data_factory_name,
                                                     pipeline_name=pipeline)
        print(f"Started pipeline: {pipeline}")
        run_ids_per_pipeline[run_result.run_id] = pipeline

    print("Waiting for pipelines to complete")
    status_per_run = {run_id: "running" for run_id in run_ids_per_pipeline.keys()}
    while True:
        for run_id in run_ids_per_pipeline.keys():
            if status_per_run[run_id]:
                pipeline_run = adf_client.pipeline_runs.get(resource_group_name=azure_config.resource_group,
                                                            factory_name=data_factory_name,
                                                            run_id=run_id)
                status = pipeline_run.status
                if status == "Succeeded" or status == "Failed":
                    print(f"Pipeline '{run_ids_per_pipeline[run_id]}' completed with status {status}")
                    status_per_run[run_id] = ""
                else:
                    status_per_run[run_id] = status
        remaining_runs = len([v for v in status_per_run.values() if v])
        print(f"Remaining pipelines that are running: {remaining_runs}")
        if remaining_runs == 0:
            break
        time.sleep(30)

    utcnow = datetime.now(timezone.utc)
    filter_params = RunFilterParameters(last_updated_after=utcnow - timedelta(days=1),
                                        last_updated_before=utcnow + timedelta(days=1))
    for run_id, pipeline in run_ids_per_pipeline.items():
        query_response = adf_client.activity_runs.query_by_pipeline_run(resource_group_name=azure_config.resource_group,
                                                                        factory_name=data_factory_name,
                                                                        run_id=run_id,
                                                                        filter_parameters=filter_params)
        run_status = query_response.value[0]
        print(f"Status for pipeline {pipeline}: {run_status.status}")
        if run_status.status == 'Succeeded':
            print(f"\tNumber of bytes read: {run_status.output['dataRead']}")
            print(f"\tNumber of bytes written: {run_status.output['dataWritten']}")
            print(f"\tCopy duration: {run_status.output['copyDuration']}")
        else:
            print(f"\tErrors: {run_status.error['message']}")

    print("All pipelines completed. Deleting data factory.")
    adf_client.factories.delete(azure_config.resource_group, data_factory_name)


def extract_access_tokens(text: str) -> Dict[str, str]:
    """
    Parses the given text for https URLs with an attached access token. Returns a dictionary mapping from
    file to file with access token, like `knee.tar.gz` -> `?AWSAccessKeyId=...`
    :param text: The text with https URLs
    :return:
    """
    result: Dict[str, str] = {}
    for match in re.finditer(r'(https://fastmri-dataset.s3.amazonaws.com/)([._a-zA-Z0-9]+)(\?[a-zA-Z0-9=&%]+)', text):
        file = match.group(2)
        token = match.group(3)
        if file in result:
            raise ValueError(f"Input file contains multiple entries for file {file}")
        print(f"Found access token for {file}: {token}")
        result[file] = token
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates an Azure Data Factory to download FastMRI data and store in'
                                                 'Azure blob storage')
    parser.add_argument(
        '--curl',
        dest='curl',
        action='store',
        type=str,
        required=True)
    parser.add_argument(
        '--connection_string',
        dest='connection_string',
        action='store',
        type=str,
        required=True)
    parser.add_argument(
        '--location',
        dest='location',
        action='store',
        type=str,
        required=True)
    known_args, unknown_args = parser.parse_known_args()
    curl_file = Path(known_args.curl)
    if not curl_file.is_file():
        raise FileNotFoundError(f"File not found: {curl_file}")
    files_and_tokens = extract_access_tokens(curl_file.read_text())
    any_files_missing = False
    for _, files in files_to_download:
        for f in files:
            source_file = f[0] if isinstance(f, tuple) else f
            if source_file not in files_and_tokens:
                any_files_missing = True
                print(f"No token found in the curl file for {source_file}")
    if any_files_missing:
        exit(1)
    create_datafactory_and_run(files_and_tokens=files_and_tokens,
                               connection_string=known_args.connection_string,
                               location=known_args.location)

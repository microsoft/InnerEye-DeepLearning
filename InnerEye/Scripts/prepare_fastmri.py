#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import List, Union

from azure.identity import ClientSecretCredential, InteractiveBrowserCredential
from azure.mgmt.datafactory import DataFactoryManagementClient
from azure.mgmt.datafactory.models import (AzureBlobStorageLinkedService,
                                           AzureBlobStorageLocation,
                                           BinaryDataset,
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
                                           SecureString)
from azure.mgmt.resource import ResourceManagementClient
from msrestazure.azure_active_directory import InteractiveCredentials, ServicePrincipalCredentials

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.secrets_handling import SecretsHandling
from InnerEye.Common import fixed_paths

# The Azure blob container that will hold all downloaded results
TARGET_CONTAINER = "datasets"
# The folder in the TARGET_CONTAINER that holds all the uncompressed files that were downloaded.
TARGET_FOLDER_UNCOMPRESSED = "fastmri_compressed"


def get_msrest_auth(azure_config: AzureConfig) -> Union[InteractiveCredentials, ServicePrincipalCredentials]:
    secrets_handler = SecretsHandling(project_root=azure_config.project_root)
    application_key = secrets_handler.get_secret_from_environment(fixed_paths.SERVICE_PRINCIPAL_KEY,
                                                                  allow_missing=True)
    if not application_key:
        logging.warning("Unable to retrieve the key for the Service Principal authentication "
                        f"(expected in environment variable '{fixed_paths.SERVICE_PRINCIPAL_KEY}' or YAML). "
                        f"Switching to interactive login.")
        return InteractiveCredentials()

    return ServicePrincipalCredentials(
        tenant=azure_config.tenant_id,
        client_id=azure_config.application_id,
        secret=application_key)


def get_azure_identity_auth(azure_config: AzureConfig) -> Union[InteractiveBrowserCredential, ClientSecretCredential]:
    secrets_handler = SecretsHandling(project_root=azure_config.project_root)
    application_key = secrets_handler.get_secret_from_environment(fixed_paths.SERVICE_PRINCIPAL_KEY,
                                                                  allow_missing=True)
    if not application_key:
        logging.warning("Unable to retrieve the key for the Service Principal authentication "
                        f"(expected in environment variable '{fixed_paths.SERVICE_PRINCIPAL_KEY}' or YAML). "
                        f"Switching to interactive login.")
        return InteractiveBrowserCredential()

    return ClientSecretCredential(
        tenant_id=azure_config.tenant_id,
        client_id=azure_config.application_id,
        client_secret=application_key)


def create_datafactory_and_run(aws_access_token: str,
                               connection_string: str,
                               is_unittest: bool = False) -> None:
    """
    Builds an Azure Data Factory to download the FastMRI dataset from AWS, and places them in Azure Blob Storage.
    :param is_unittest: If True, download a small tar.gz file from github. If False, download the "real" fastMRI
    datafiles from AWS.
    :param aws_access_token: The access token for accessing the FastMRI data on AWS.
    :param connection_string: The connection string of the Azure storage where the downloaded data should be stored.
    """

    azure_config = AzureConfig.from_yaml(yaml_file_path=fixed_paths.SETTINGS_YAML_FILE,
                                         project_root=fixed_paths.repository_root_directory())

    # The data factory name. It must be globally unique.
    data_factory_name = "fastmri-copy-data"

    # Get either the Service Principal authentication, if those are set already, or use interactive auth in the browser
    msrest_auth = get_msrest_auth(azure_config)

    print(f"Retrieving resource group {azure_config.resource_group}")
    resource_client = ResourceManagementClient(msrest_auth, azure_config.subscription_id)
    resource_group = resource_client.resource_groups.get(resource_group_name=azure_config.resource_group)

    # Create a data factory
    azureid_auth = get_azure_identity_auth(azure_config)
    adf_client = DataFactoryManagementClient(azureid_auth, azure_config.subscription_id)
    df_resource = Factory(location=resource_group.location)
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

    def download_and_uncompress(source_file: str, target_folder: str) -> List[str]:
        """
        Downloads a file from AWS and stores them in blob storage in its compressed form.
        Also, downloads the same file from AWS, uncompresses it, and writes the results to the given folder
        in blob storage.
        :param source_file: The name of the .tar.gz or .tar file to download, without any access tokens.
        :param target_folder: The folder in the target storage account
        :return:
        """
        if is_unittest:
            http_source = HttpServerLocation(relative_url="gulpjs/gulp/archive/v3.9.1.tar.gz")
        else:
            http_source = HttpServerLocation(relative_url=f"{source_file}{aws_access_token}")
        source_file_cleaned = source_file.replace(".", "_")
        # A dataset that reads the files from AWS as-is
        source_compressed = BinaryDataset(linked_service_name=linked_http,
                                          location=http_source)
        source_compressed_name = f"read {source_file_cleaned}"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=source_compressed_name,
                                             dataset=DatasetResource(properties=source_compressed))
        # A dataset that reads the files from AWS and uncompresses on-the-fly
        if source_file.endswith(".tar.gz"):
            compression = DatasetTarGZipCompression()
        elif source_file.endswith(".tar"):
            compression = DatasetTarCompression()
        else:
            raise ValueError(f"Unable to determine compression for file {source_file}")
        source_uncompressed = BinaryDataset(linked_service_name=linked_http,
                                            location=http_source,
                                            compression=compression)
        source_uncompressed_name = f"read {source_file_cleaned} and uncompress"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=source_uncompressed_name,
                                             dataset=DatasetResource(properties=source_uncompressed))
        # The sink for downloading the datasets as-is (compressed)
        dest_compressed = BinaryDataset(linked_service_name=linked_blob_storage,
                                        location=AzureBlobStorageLocation(file_name=source_file,
                                                                          container=TARGET_CONTAINER,
                                                                          folder_path=TARGET_FOLDER_UNCOMPRESSED))
        dest_compressed_name = f"save {source_file_cleaned} compressed"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=dest_compressed_name,
                                             dataset=DatasetResource(properties=dest_compressed))
        # The sink for downloading the datasets uncompressed
        final_dataset = BinaryDataset(linked_service_name=linked_blob_storage,
                                      location=AzureBlobStorageLocation(container=TARGET_CONTAINER,
                                                                        folder_path=target_folder))
        final_name = f"save {source_file_cleaned} uncompressed"
        adf_client.datasets.create_or_update(resource_group_name=azure_config.resource_group,
                                             factory_name=data_factory_name,
                                             dataset_name=final_name,
                                             dataset=DatasetResource(properties=final_dataset))
        # Copying from compressed source to compressed destination
        pipeline1 = f"download {source_file_cleaned}"
        download = CopyActivity(name=f"download {source_file_cleaned}",
                                inputs=[DatasetReference(reference_name=source_compressed_name)],
                                outputs=[DatasetReference(reference_name=dest_compressed_name)],
                                source=HttpSource(),
                                sink=BlobSink())
        adf_client.pipelines.create_or_update(resource_group_name=azure_config.resource_group,
                                              factory_name=data_factory_name,
                                              pipeline_name=pipeline1,
                                              pipeline=PipelineResource(activities=[download]))
        # Copying from compressed source to uncompressed destination
        uncompress = CopyActivity(name=f"uncompress {source_file_cleaned}",
                                  inputs=[DatasetReference(reference_name=source_uncompressed_name)],
                                  outputs=[DatasetReference(reference_name=final_name)],
                                  source=HttpSource(),
                                  sink=BlobSink())
        pipeline2 = f"uncompress {source_file_cleaned}"
        adf_client.pipelines.create_or_update(resource_group_name=azure_config.resource_group,
                                              factory_name=data_factory_name,
                                              pipeline_name=pipeline2,
                                              pipeline=PipelineResource(activities=[uncompress]))
        return [pipeline1, pipeline2]

    # Mapping different of the raw tar.gz files to separate dataset folders.
    # First tuple item is the target folder in blob storage, then comes a list of raw files to extract into that folder.
    files_to_download = [
        ("knee_singlecoil", ["knee_singlecoil_train.tar.gz",
                             "knee_singlecoil_val.tar.gz",
                             "knee_singlecoil_test_v2.tar.gz",
                             "knee_singlecoil_challenge.tar.gz"]),
        ("knee_multicoil", ["knee_multicoil_train.tar.gz",
                            "knee_multicoil_val.tar.gz",
                            "knee_multicoil_test_v2.tar.gz",
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

    if is_unittest:
        files_to_download = [("antonsctest", ["foo.tar.gz", "bar.tar"])]

    all_pipelines = []
    print("Creating pipelines:")
    for target_folder, files in files_to_download:
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
        time.sleep(10)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Creates an Azure Data Factory to download FastMRI data and store in'
                                                 'Azure blob storage')
    parser.add_argument(
        '--aws_token',
        dest='aws_token',
        action='store',
        type=str,
        required=True)
    parser.add_argument(
        '--connection_string',
        dest='connection_string',
        action='store',
        type=str,
        required=True)
    known_args, unknown_args = parser.parse_known_args()
    aws_access_token = known_args.aws_token
    connection_string = known_args.connection_string
    create_datafactory_and_run(aws_access_token=aws_access_token,
                               connection_string=connection_string)

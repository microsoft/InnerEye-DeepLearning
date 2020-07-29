#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import blobxfer
import blobxfer.models.azure as azmodels
from blobxfer.api import AzureStorageCredentials, ConcurrencyOptions, DownloadOptions, GeneralOptions, SkipOnOptions, \
    UploadOptions
from blobxfer.models.options import FileProperties, Timeout, VectoredIo
from blobxfer.models.upload import VectoredIoDistributionMode

from InnerEye.Azure.azure_util import storage_account_from_full_name, to_azure_friendly_container_path

# Azure storage is extremely talkative, printing out each client request (thousands of them)
logger = logging.getLogger('azure.storage')
logger.setLevel(logging.WARNING)
# Blobxfer also prints at single line per file at least.
logger = logging.getLogger('blobxfer')
logger.setLevel(logging.WARNING)


@dataclass
class BlobXFerConfig:
    """
    Class to hold Bloxfer configurations and helpers functions to download and
    upload functions to Azure.
    """
    account_name: str
    account_key: str
    concurrency: ConcurrencyOptions
    timeout: Timeout
    general: GeneralOptions
    file_properties: FileProperties
    skipon_options: SkipOnOptions

    @staticmethod
    def create_default(account: str, account_key: str) -> BlobXFerConfig:
        """
        Returns the default configuration.

        :param account: Name of the Azure storage account
        :param account_key: Key to this storage account
        :return: default Blobxferconfig
        """
        concurrency = ConcurrencyOptions(crypto_processes=2, md5_processes=2, disk_threads=16,
                                         transfer_threads=16, action=1)
        timeout = Timeout(connect=20, read=60, max_retries=3)
        general = GeneralOptions(concurrency, progress_bar=False, verbose=False, timeout=timeout, quiet=True)
        file_properties = FileProperties(attributes=False, cache_control=None, content_type=None,
                                         lmt=False, md5=None)
        skipon_options = SkipOnOptions(filesize_match=True, lmt_ge=True, md5_match=True)
        return BlobXFerConfig(
            account_name=storage_account_from_full_name(account),
            account_key=account_key,
            concurrency=concurrency,
            timeout=timeout,
            general=general,
            file_properties=file_properties,
            skipon_options=skipon_options
        )

    def get_download_options(self, num_folders_to_strip: int = 0) -> DownloadOptions:
        """
        Returns a BloxFer DownloadOptions object.

        :param num_folders_to_strip: The filenames will be stripped off their leading directories, up to this level e.g.
        if original path is 'container/foo/1.txt' and number_folders_to_strip is 2, and destination folder is 'bar',
        the downloaded file will be 'bar/1.txt'
        """
        return DownloadOptions(check_file_md5=True,
                               chunk_size_bytes=4194304,
                               delete_extraneous_destination=False,
                               delete_only=False,
                               max_single_object_concurrency=8,
                               mode=azmodels.StorageModes.Auto,
                               overwrite=True,
                               recursive=True,
                               rename=False,
                               restore_file_properties=self.file_properties,
                               rsa_private_key=None,
                               strip_components=num_folders_to_strip)

    def get_upload_options(self, num_folders_to_strip: int = 0) -> UploadOptions:
        """
        Returns a UploadOptions object.

        :param num_folders_to_strip: The filenames will be stripped off their leading directories, up to this level e.g.
        if original path is 'container/foo/1.txt' and number_folders_to_strip is 2, and destination folder is 'bar',
        the downloaded file will be 'bar/1.txt'
        """
        return UploadOptions(
            access_tier=None,
            one_shot_bytes=33554432,
            rsa_public_key=None,
            stdin_as_page_blob_size=0,
            store_file_properties=self.file_properties,
            vectored_io=VectoredIo(
                stripe_chunk_size_bytes=0,
                distribution_mode=VectoredIoDistributionMode.Disabled
            ),
            chunk_size_bytes=4194304,
            delete_extraneous_destination=False,
            delete_only=False,
            mode=azmodels.StorageModes.Auto,
            overwrite=True,
            recursive=True,
            rename=False,
            strip_components=num_folders_to_strip
        )

    def get_credentials(self) -> AzureStorageCredentials:
        credentials = AzureStorageCredentials(self.general)
        credentials.add_storage_account(self.account_name, self.account_key, endpoint="core.windows.net")
        return credentials


def download_blobs(account: str, account_key: str, blobs_root_path: str, destination: Path,
                   is_file: bool = False, config: Optional[BlobXFerConfig] = None) -> Path:
    """
    Download a given set of files in Azure blob storage to the local destination path, via blobxfer.
    :param account: The name of the storage account to access the files.
    :param account_key: The key for the storage account.
    :param blobs_root_path: The path of the files that should be downloaded. This must be in format
    'container/file_prefix/', ending with a slash (will be added if not provided and is_file is False).
    :param destination: The destination folder for the copied files on the local machine.
    :param is_file: If True then only a single file is required to be downloaded
    :param config: BlobXFerConfig to use for download configuration, use default presets if None.
    The filenames will be stripped off their leading directories, up to the level given by blobs_root_path.
    For example, if blobs_root_path is 'container/foo/'
    and contains a file 'container/foo/1.txt', and destination is 'bar', the downloaded file will be 'bar/1.txt'
    """
    if not config:
        config = BlobXFerConfig.create_default(account=account, account_key=account_key)

    start_time = time.time()
    # the account name can be an Azure Resource ID so extract the name from it if this is the case
    logging.info(f"Downloading '{blobs_root_path}' from storage account {config.account_name} to ${destination}")
    blobs_root_path = to_azure_friendly_container_path(Path(blobs_root_path))
    if not (blobs_root_path.endswith("/") or is_file):
        blobs_root_path += "/"
    blobs_root_path_dirs = blobs_root_path.rstrip("/").split("/")
    num_folders_to_strip = len(blobs_root_path_dirs) - 1

    blobs_path_without_container = "/".join(blobs_root_path_dirs[1:])
    logging.info(f"Cleaned download path: '{blobs_root_path}' from storage account {config.account_name}")

    download = config.get_download_options(num_folders_to_strip)
    local_path = blobxfer.api.LocalDestinationPath(str(destination))
    # noinspection PyTypeChecker
    download_spec = blobxfer.api.DownloadSpecification(download, config.skipon_options, local_path)
    source = blobxfer.api.AzureSourcePath()
    source.add_path_with_storage_account(blobs_root_path, config.account_name)
    if not is_file:
        source.add_includes([f"{blobs_path_without_container}/*"])
    download_spec.add_azure_source_path(source)
    # noinspection PyTypeChecker
    downloader = blobxfer.api.Downloader(config.general, config.get_credentials(), download_spec)
    downloader.start()
    elapsed = time.time() - start_time
    logging.info(f"Finished downloading in {elapsed:0.2f}sec.")

    if is_file:
        destination = destination / Path(blobs_root_path).name
        if destination.exists():
            return destination
        raise ValueError(f"Unable to download {blobs_root_path} from "
                         f"storage account {config.account_name} to {destination}")
    else:
        return destination

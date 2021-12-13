#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from health_azure import DatasetConfig
from health_azure.utils import get_workspace


def mount_dataset(dataset_id: str) -> str:
    ws = get_workspace()
    target_folder = "/tmp/datasets/"
    dataset = DatasetConfig(name=dataset_id, target_folder=target_folder, use_mounting=True)
    dataset_mount_folder, mount_ctx = dataset.to_input_dataset_local(ws)
    mount_ctx.start()
    assert next(dataset_mount_folder.iterdir()), "Mounted data folder is empty"
    return str(dataset_mount_folder)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, dest='dataset_id',
                        help='Name of the Azure dataset e.g. PANDA or TCGA-CRCk')
    args = parser.parse_args()
    mount_dataset(args.dataset_id)

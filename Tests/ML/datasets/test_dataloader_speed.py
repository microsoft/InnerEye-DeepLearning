#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import time
from io import StringIO

import pandas as pd
import pytest

from InnerEye.Common.common_util import is_windows
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.utils import ml_util


@pytest.mark.parametrize("num_dataload_workers", [0, 1])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.skipif(is_windows(),
                    reason="This test runs fine on local Windows boxes, but leads to odd timeouts in Azure")
def test_dataloader_speed(test_output_dirs: OutputFolderForTests,
                          num_dataload_workers: int,
                          shuffle: bool) -> None:
    """
    Test how dataloaders work when using multiple processes.
    """
    ml_util.set_random_seed(0)
    # The dataset should only contain the file name stem, without extension.
    csv_string = StringIO("""subject,channel,path,value,scalar1
S1,image,4be9beed-5861-fdd2-72c2-8dd89aadc1ef
S1,label,,True,1.0
S2,image,6ceacaf8-abd2-ffec-2ade-d52afd6dd1be
S2,label,,True,2.0
S3,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S3,label,,False,3.0
S4,image,61bc9d73-9fbb-bd7d-c06b-eeffbafabcc4
S4,label,,False,3.0
""")
    args = ScalarModelBase(image_channels=[],
                           label_channels=["label"],
                           label_value_column="value",
                           non_image_feature_channels=["label"],
                           numerical_columns=["scalar1"],
                           num_dataload_workers=num_dataload_workers,
                           num_dataset_reader_workers=num_dataload_workers,
                           avoid_process_spawn_in_data_loaders=True,
                           should_validate=False)
    dataset = ScalarDataset(args, data_frame=pd.read_csv(csv_string, dtype=str))
    assert len(dataset) == 4
    num_epochs = 2
    total_start_time = time.time()
    loader = dataset.as_data_loader(shuffle=shuffle, batch_size=1)
    # The order in which items are expected in each epoch, when applying shuffling, and using 1 dataloader worker
    # This was determined before making any changes to the dataloader logic
    # (that is, when the as_data_loader method returns an instance of DataLoader, rather than RepeatDataLoader)
    expected_item_order = [
        ["S2", "S1", "S4", "S3"],
        ["S4", "S3", "S1", "S2"],
    ]
    for epoch in range(num_epochs):
        actual_item_order = []
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()
        item_start_time = time.time()
        for i, item_dict in enumerate(loader):
            item_load_time = time.time() - item_start_time
            item = ScalarItem.from_dict(item_dict)
            # noinspection PyTypeChecker
            sample_id = item.metadata[0].id  # type: ignore
            print(f"Loading item {i} with ID = {sample_id} in {item_load_time:0.8f} sec")
            if shuffle:
                actual_item_order.append(sample_id)
            else:
                assert sample_id == f"S{i + 1}"
            if not (epoch == 0 and i == 0):
                assert item_load_time < 0.1, f"We should only see significant item load times in the first batch " \
                                             f"of the first epoch, but got loading time of {item_load_time:0.2f} sec" \
                                             f" in epoch {epoch} batch {i}"
            # Sleep a bit so that the worker process can fill in items
            if num_dataload_workers > 0:
                time.sleep(0.05)
            item_start_time = time.time()
        if shuffle and num_dataload_workers == 1:
            assert actual_item_order == expected_item_order[epoch], f"Item in wrong order for epoch {epoch}"
        total_epoch_time = time.time() - epoch_start_time
        print(f"Total time for epoch {epoch}: {total_epoch_time} sec")
    total_time = time.time() - total_start_time
    print(f"Total time for all epochs: {total_time} sec")

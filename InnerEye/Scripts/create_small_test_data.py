#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
r"""
Script to create smaller and broader test data than the images found in
InnerEye\Tests\ML\test_data\train_and_test_data. We provide multiple non-overlapping labels, and
an additional subject.
To run from the repository root:
    conda activate InnerEye
    export PYTHONPATH=`pwd`
    python InnerEye/scripts/create_small_test_data.py
"""
from pathlib import Path
import numpy as np
from InnerEye.ML.utils import io_util
from InnerEye.ML.utils.io_util import ImageHeader

XY_DIMENSION = 25
Z_DIMENSION = 10

def create_small_train_and_test_data(output_dir: Path) -> None:
    """
    """
    for id in ["id1", "id2", "id3"]:
        for channel in ["channel1", "channel2"]:
            image = np.random.random_sample((Z_DIMENSION, XY_DIMENSION, XY_DIMENSION))
            image = np.array((image + 1) * 255).astype(int)
            header = ImageHeader(origin=(1, 1, 1), direction=(1, 0, 0, 0, 1, 0, 0, 0, 1), spacing=(1, 1, 1))
            io_util.store_as_nifti(image, header, (output_dir / (id + channel + ".nii.gz")).absolute, np.ubyte)


def main() -> None:
    output_dir = Path("train_and_test_data", "small")
    output_dir.mkdir(parents=True, exist_ok=True)
    create_small_train_and_test_data(output_dir)


if __name__ == "__main__":
    main()

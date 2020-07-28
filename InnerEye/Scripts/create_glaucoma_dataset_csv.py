#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
"""
This script takes a folder with images and builds a dataset.csv with subject IDs, image file paths and labels
by parsing the image filenames. This script is meant to work with the glaucoma dataset in the
sample classification task. For details on creating classification datasets,
see https://github.com/microsoft/InnerEye-createdataset
"""

import sys
from pathlib import Path
from typing import Tuple

# each row in the dataset will contain 5 values: subject ID, channel, file path, label, and date
DatasetRow = Tuple[str, str, str, str, str]


def parse(file_name: Path) -> DatasetRow:
    """
    Takes a filename and parses the name to find the subject ID, label and date. The filename has the form
    <type>-<subjectID>-<date>-<eye>.npy.
    - type is either Normal when no glaucoma is present, or POAG if glaucoma is present.
    - subjectID is a 6 digit numeric ID that uniquely identifies each subject
    - date, in the form yyyy-mm-dd
    - eye is either OD or OS

    :param file_name: File naem to be parsed.
    :return: A tuple of 5 strings with the values of subject ID, channel, path, label and date for this file.
    """
    parts = file_name.stem.split("-")
    assert len(parts) == 6
    if parts[0].startswith("POAG"):
        label = "True"
    elif parts[0].startswith("Normal"):
        label = "False"
    else:
        raise ValueError(f"Invalid prefix: {parts[0]}")
    eye = parts[5]
    subject_id_and_eye = parts[1] + "-" + eye
    date = "-".join(parts[2:5])
    return subject_id_and_eye, "image", file_name.name, label, date


if __name__ == '__main__':
    if not len(sys.argv) == 2:
        raise ValueError(f"{sys.argv[0]} takes exactly one argument for dataset path")
    dataset_dir = Path(sys.argv[1])
    numpy_files = list(dataset_dir.glob("*.npy"))
    header = ["subject,channel,filePath,label,date"]
    lines = [",".join(parse(f)) for f in numpy_files]
    out_file = dataset_dir / "dataset.csv"
    out_file.write_text("\n".join(header + lines))

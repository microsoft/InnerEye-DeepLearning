#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict

import papermill
import nbformat
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter


def generate_notebook(notebook_path: Path, notebook_params: Dict, result_path: Path) -> None:
    print(f"Writing report to {result_path}")
    papermill.execute_notebook(input_path=str(notebook_path),
                               output_path=str(result_path),
                               parameters=notebook_params,
                               progress_bar=False)
    print(f"Running conversion to html for {result_path}")
    with open(str(result_path)) as f:
        notebook = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        (body, resources) = html_exporter.from_notebook_node(notebook)
        write_file = FilesWriter()
        write_file.write(
            output=body,
            resources=resources,
            notebook_name=result_path.resolve().stem
        )

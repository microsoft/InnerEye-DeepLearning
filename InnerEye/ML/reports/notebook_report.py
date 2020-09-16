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
from pandas import DataFrame
import matplotlib.pyplot as plt
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns


def boxplot_per_structure(df: DataFrame, column_name: str,
                          title: str) -> None:
    """
    Creates a box-and-whisker plot for a score per structure. Structures are on the x-axis,
    box plots are drawn vertically. The plot is created in the currently active figure or subplot.
    """
    structure = MetricsFileColumns.Structure.value
    dice_numeric = column_name
    structure_series = df[structure]
    unique_structures = structure_series.unique()
    dice_per_structure = [df[dice_numeric][structure_series == s] for s in unique_structures]
    # If there are only single entries per structure, do not generate a box plot
    if all([len(dps) == 1 for dps in dice_per_structure]):
        return

    plt.title(title)
    plt.boxplot(dice_per_structure, labels=unique_structures)
    plt.xlabel("Structure")
    plt.ylabel(column_name)
    plt.xticks(rotation=75)
    plt.grid()

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
        write_file.build_directory = str(result_path.parent)
        write_file.write(
            output=body,
            resources=resources,
            notebook_name=str(result_path.resolve().stem)
        )

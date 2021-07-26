#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import codecs
import nbformat
import papermill
import pickle
from IPython.display import HTML, Markdown, display
from nbconvert import HTMLExporter
from nbconvert.writers import FilesWriter

from InnerEye.Common import fixed_paths
from InnerEye.ML.scalar_config import ScalarModelBase


REPORT_PREFIX = "report"
REPORT_IPYNB_SUFFIX = ".ipynb"
REPORT_HTML_SUFFIX = ".html"
reports_folder = "reports"


def get_ipynb_report_name(report_type: str) -> str:
    """
    Constructs the name of the report (as an ipython notebook).
    :param report_type: suffix describing the report, added to the filename
    :return:
    """
    return f"{REPORT_PREFIX}_{report_type}{REPORT_IPYNB_SUFFIX}"


def get_html_report_name(report_type: str) -> str:
    """
    Constructs the name of the report (as an html file).
    :param report_type: suffix describing the report, added to the filename
    :return:
    """
    return f"{REPORT_PREFIX}_{report_type}{REPORT_HTML_SUFFIX}"


def str_or_empty(p: Union[None, str, Path]) -> str:
    return str(p) if p else ""


def print_header(message: str, level: int = 2) -> None:
    """
    Displays a message, and afterwards repeat it as Markdown with the given indentation level (level=1 is the
    outermost, `# Foo`.
    :param message: The header string to display.
    :param level: The Markdown indentation level. level=1 for top level, level=3 for `### Foo`
    """
    prefix = "#" * level
    display(Markdown(f"{prefix} {message}"))


def print_table(rows: Sequence[Sequence[str]], header: Optional[Sequence[str]] = None) -> None:
    """
    Displays the provided content in a formatted HTML table, with optional column headers.
    :param rows: List of rows, where each row is a list of string-valued cell contents.
    :param header: List of column headers. If given, this special first row is rendered with emphasis.
    """
    if any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError("All rows in the table should have the same length")
    if header and len(header) != len(rows[0]):
        raise ValueError("Table header and rows should have the same length")
    import pandas as pd
    df = pd.DataFrame(data=rows, columns=header)
    display(HTML(df.to_html(index=False)))


def generate_notebook(template_notebook: Path, notebook_params: Dict, result_notebook: Path) -> Path:
    """
    Generates a notebook report as jupyter notebook and html page
    :param template_notebook: path to template notebook
    :param notebook_params: parameters for the notebook
    :param result_notebook: the path for the executed notebook
    :return: returns path to the html page
    """
    print(f"Writing report to {result_notebook}")
    papermill.execute_notebook(input_path=str(template_notebook),
                               output_path=str(result_notebook),
                               parameters=notebook_params,
                               progress_bar=False,
                               # Unit tests often fail with cell timeouts when default of 4 is used.
                               iopub_timeout=10)
    return convert_to_html(result_notebook)


def convert_to_html(result_notebook: Path) -> Path:
    """
    :param result_notebook: The path to the result notebook
    :return: Path with output extension
    """
    print(f"Running conversion to HTML for {result_notebook}")
    with result_notebook.open() as f:
        notebook = nbformat.read(f, as_version=4)
        html_exporter = HTMLExporter()
        html_exporter.exclude_input = True
        (body, resources) = html_exporter.from_notebook_node(notebook)
        write_file = FilesWriter()
        write_file.build_directory = str(result_notebook.parent)
        write_file.write(
            output=body,
            resources=resources,
            notebook_name=str(result_notebook.stem)
        )
    return result_notebook.with_suffix(resources['output_extension'])


def generate_segmentation_notebook(result_notebook: Path,
                                   train_metrics: Optional[Path] = None,
                                   val_metrics: Optional[Path] = None,
                                   test_metrics: Optional[Path] = None) -> Path:
    """
    Creates a reporting notebook for a segmentation model, using the given training, validation, and test set metrics.
    Returns the report file after HTML conversion.
    """

    notebook_params = \
        {
            'innereye_path': str(fixed_paths.repository_root_directory()),
            'train_metrics_csv': str_or_empty(train_metrics),
            'val_metrics_csv': str_or_empty(val_metrics),
            'test_metrics_csv': str_or_empty(test_metrics),
        }
    template = Path(__file__).absolute().parent / "segmentation_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)


def generate_classification_notebook(result_notebook: Path,
                                     config: ScalarModelBase,
                                     train_metrics: Optional[Path] = None,
                                     val_metrics: Optional[Path] = None,
                                     test_metrics: Optional[Path] = None) -> Path:
    """
    Creates a reporting notebook for a classification model, using the given training, validation, and test set metrics.
    Returns the report file after HTML conversion.
    """

    notebook_params = \
        {
            'innereye_path': str(fixed_paths.repository_root_directory()),
            'train_metrics_csv': str_or_empty(train_metrics),
            'val_metrics_csv': str_or_empty(val_metrics),
            'test_metrics_csv': str_or_empty(test_metrics),
            "config": codecs.encode(pickle.dumps(config), "base64").decode(),
            "is_crossval_report": False
        }
    template = Path(__file__).absolute().parent / "classification_crossval_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)


def generate_classification_crossval_notebook(result_notebook: Path,
                                              config: ScalarModelBase,
                                              crossval_metrics: Path) -> Path:
    """
    Creates a reporting notebook for a classification model, using the given training, validation, and test set metrics.
    Returns the report file after HTML conversion.
    """

    notebook_params = \
        {
            'innereye_path': str(fixed_paths.repository_root_directory()),
            'train_metrics_csv': "",
            'val_metrics_csv': str_or_empty(crossval_metrics),
            'test_metrics_csv': "",
            "config": codecs.encode(pickle.dumps(config), "base64").decode(),
            "is_crossval_report": True
        }
    template = Path(__file__).absolute().parent / "classification_crossval_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)


def generate_classification_multilabel_notebook(result_notebook: Path,
                                                config: ScalarModelBase,
                                                train_metrics: Optional[Path] = None,
                                                val_metrics: Optional[Path] = None,
                                                test_metrics: Optional[Path] = None) -> Path:
    """
    Creates a reporting notebook for a multilabel classification model, using the given training, validation,
    and test set metrics. This report adds metrics specific to the multilabel task, and is meant to be used in
    addition to the standard report created for all classification models.
    Returns the report file after HTML conversion.
    """

    notebook_params = \
        {
            'innereye_path': str(fixed_paths.repository_root_directory()),
            'train_metrics_csv': str_or_empty(train_metrics),
            'val_metrics_csv': str_or_empty(val_metrics),
            'test_metrics_csv': str_or_empty(test_metrics),
            "config": codecs.encode(pickle.dumps(config), "base64").decode()
        }
    template = Path(__file__).absolute().parent / "classification_multilabel_report.ipynb"
    return generate_notebook(template,
                             notebook_params=notebook_params,
                             result_notebook=result_notebook)

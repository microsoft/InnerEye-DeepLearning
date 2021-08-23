#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEye.Common.common_util import ModelProcessing, get_best_epoch_results_path
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.metrics_constants import LoggingColumns
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.configs.classification.CovidModel import CovidModel
from InnerEye.ML.model_testing import MODEL_OUTPUT_CSV


def test_generate_custom_report(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test that the Covid model report is generated correctly
    (especially when there are NaN values in the hierarchical task).
    """

    model = CovidModel()
    model.set_output_to(test_output_dirs.root_dir)
    report_dir = test_output_dirs.root_dir / "reports"
    report_dir.mkdir()

    train_csv_path = model.outputs_folder / get_best_epoch_results_path(mode=ModelExecutionMode.TRAIN,
                                                                             model_proc=ModelProcessing.DEFAULT) \
                     / MODEL_OUTPUT_CSV
    train_csv_path.parent.mkdir(parents=True)
    train_csv_path.write_text(f"""{LoggingColumns.Patient.value},{LoggingColumns.Hue.value},{LoggingColumns.Label.value},{LoggingColumns.ModelOutput.value},{LoggingColumns.CrossValidationSplitIndex.value}
1,CVX0,1,0.7,-1
1,CVX1,0,0.1,-1
1,CVX2,0,0.1,-1
1,CVX3,0,0.1,-1
2,CVX0,0,0.1,-1
2,CVX1,1,0.7,-1
2,CVX2,0,0.1,-1
2,CVX3,0,0.1,-1
3,CVX0,0,0.7,-1
3,CVX1,0,0.1,-1
3,CVX2,1,0.1,-1
3,CVX3,0,0.1,-1
4,CVX0,0,0.0,-1
4,CVX1,0,1.0,-1
4,CVX2,0,0.0,-1
4,CVX3,1,0.0,-1
5,CVX0,0,0.0,-1
5,CVX1,0,0.0,-1
5,CVX2,1,1.0,-1
5,CVX3,0,0.0,-1
6,CVX0,0,0.0,-1
6,CVX1,1,1.0,-1
6,CVX2,0,0.0,-1
6,CVX3,0,0.0,-1
""")

    report_path = model.generate_custom_report(report_dir=report_dir, model_proc=ModelProcessing.DEFAULT)
    report_text = report_path.read_text()

    assert report_text == f"""{ModelExecutionMode.TRAIN.value}
CVX03vs12 Accuracy: 0.6667
CVX0vs3 Accuracy: 1.0000
Warning: CVX0vs3 accuracy was computed skipping 1 NaN model outputs.
CVX1vs2 Accuracy: 0.7500
Multiclass Accuracy: 0.6667

"""

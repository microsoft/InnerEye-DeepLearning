#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from InnerEye.Azure.azure_config import AzureConfig
from InnerEye.Azure.azure_runner import create_runner_parser, parse_args_and_add_yaml_variables
from InnerEye.Azure.azure_util import download_outputs_from_run
from InnerEye.ML.common import ModelExecutionMode
from InnerEye.ML.config import SegmentationModelBase
from InnerEye.ML.plotting import surface_distance_ground_truth_plot, segmentation_and_groundtruth_plot
from InnerEye.ML.utils import surface_distance_utils as sd_util
from InnerEye.ML.utils.config_util import ModelConfigLoader
from InnerEye.ML.utils.csv_util import get_worst_performing_outliers, load_csv
from InnerEye.ML.utils.image_util import multi_label_array_to_binary
from InnerEye.ML.utils.io_util import load_nifti_image
from InnerEye.ML.utils.metrics_constants import MetricsFileColumns
from InnerEye.ML.utils.surface_distance_utils import SurfaceDistanceConfig, SurfaceDistanceRunType


@dataclass(frozen=True)
class Segmentation:
    """
    Each individual structure segmentation (whether model prediction or human annotation) will have the properties
    structure_name (i.e. body part), subject_id and a unique path. Optionally, it may also have an associated
    annotator name and calculated Dice score, compared to ground truth.
    """
    segmentation_path: Path
    structure_name: str
    subject_id: int
    annotator: Optional[str] = None
    dice_score: Optional[float] = None


def load_predictions(run_type: SurfaceDistanceRunType, azure_config: AzureConfig, model_config: SegmentationModelBase,
                     execution_mode: ModelExecutionMode, extended_annotators: List[str], outlier_range: float
                     ) -> List[Segmentation]:
    """
    For each run type (IOV or outliers), instantiate a list of predicted Segmentations and return
    :param run_type: either "iov" or "outliers:
    :param azure_config: AzureConfig
    :param model_config: GenericConfig
    :param execution_mode: ModelExecutionMode: Either Test, Train or Val
    :param extended_annotators: List of annotators plus model_name to load segmentations for
    :param outlier_range: The standard deviation from the mean which the points have to be below
    to be considered an outlier.
    :return: list of [(subject_id, structure name and dice_scores)]
    """
    predictions = []
    if run_type == SurfaceDistanceRunType.OUTLIERS:
        first_child_run = sd_util.get_first_child_run(azure_config)
        output_dir = sd_util.get_run_output_dir(azure_config, model_config)
        metrics_path = sd_util.get_metrics_path(azure_config, model_config)

        # Load the downloaded metrics CSV as dataframe and determine worst performing outliers for the Test run
        df = load_csv(metrics_path, [MetricsFileColumns.Patient.value, MetricsFileColumns.Structure.value])
        test_run_df = df[df['mode'] == execution_mode.value]
        worst_performers = get_worst_performing_outliers(test_run_df, outlier_range, MetricsFileColumns.Dice.value,
                                                         max_n_outliers=-50)

        for (subject_id, structure_name, dice_score, _) in worst_performers:
            subject_prefix = sd_util.get_subject_prefix(model_config, execution_mode, subject_id)
            # if not already present, download data for subject
            download_outputs_from_run(
                blobs_path=subject_prefix,
                destination=output_dir,
                run=first_child_run
            )

            # check it has been downloaded
            segmentation_path = output_dir / subject_prefix / f"{structure_name}.nii.gz"
            predictions.append(Segmentation(structure_name=structure_name, subject_id=subject_id,
                                            segmentation_path=segmentation_path, dice_score=float(dice_score)))

    elif run_type == SurfaceDistanceRunType.IOV:
        subject_id = 0
        iov_dir = Path("outputs") / SurfaceDistanceRunType.IOV.value.lower()
        all_structs = model_config.class_and_index_with_background()
        structs_to_plot = [struct_name for struct_name in all_structs.keys() if struct_name not in ['background',
                                                                                                    'external']]
        for annotator in extended_annotators:
            for struct_name in structs_to_plot:
                segmentation_path = iov_dir / f"{struct_name + annotator}.nii.gz"
                if not segmentation_path.is_file():
                    logging.warning(f"No such file {segmentation_path}")
                    continue
                predictions.append(Segmentation(structure_name=struct_name, subject_id=subject_id,
                                                segmentation_path=segmentation_path, annotator=annotator))
    return predictions


def main() -> None:
    parser = create_runner_parser(SegmentationModelBase)
    parser_result = parse_args_and_add_yaml_variables(parser, fail_on_unknown_args=True)
    surface_distance_config = SurfaceDistanceConfig.parse_args()

    azure_config = AzureConfig(**parser_result.args)
    config_model = azure_config.model
    if config_model is None:
        raise ValueError("The name of the model to train must be given in the --model argument.")

    model_config = ModelConfigLoader[SegmentationModelBase]().create_model_config_from_name(
        config_model,
        overrides=parser_result.overrides
    )
    execution_mode = surface_distance_config.execution_mode

    run_mode = surface_distance_config.run_mode
    if run_mode == SurfaceDistanceRunType.IOV:
        ct_path = Path("outputs") / SurfaceDistanceRunType.IOV.value.lower() / "ct.nii.gz"
        ct = load_nifti_image(ct_path).image
    else:
        ct = None
    annotators = [annotator.strip() for annotator in surface_distance_config.annotators]
    extended_annotators = annotators + [surface_distance_config.model_name]

    outlier_range = surface_distance_config.outlier_range
    predictions = load_predictions(run_mode, azure_config, model_config, execution_mode, extended_annotators,
                                   outlier_range)
    segmentations = [load_nifti_image(Path(pred_seg.segmentation_path)) for pred_seg in predictions]
    img_shape = segmentations[0].image.shape
    # transpose spacing to match image which is transposed in io_util
    voxel_spacing = segmentations[0].header.spacing[::-1]

    overall_gold_standard = np.zeros(img_shape)
    sds_for_annotator = sd_util.initialise_surface_distance_dictionary(extended_annotators, img_shape)

    plane = surface_distance_config.plane
    output_img_dir = Path(surface_distance_config.output_img_dir)

    subject_id: Optional[int] = None
    for prediction, pred_seg_w_header in zip(predictions, segmentations):
        subject_id = prediction.subject_id
        structure_name = prediction.structure_name
        annotator = prediction.annotator
        pred_segmentation = pred_seg_w_header.image
        if run_mode == SurfaceDistanceRunType.OUTLIERS:
            try:
                ground_truth = sd_util.load_ground_truth_from_run(model_config, surface_distance_config,
                                                                  subject_id, structure_name)
            except FileNotFoundError as e:
                logging.warning(e)
                continue
        elif run_mode == SurfaceDistanceRunType.IOV:
            ground_truth = sd_util.get_annotations_and_majority_vote(model_config, annotators, structure_name)
        else:
            raise ValueError(f'Unrecognised run mode: {run_mode}. Expected either IOV or OUTLIERS')

        binary_prediction_mask = multi_label_array_to_binary(pred_segmentation, 2)[1]
        # For comparison, plot gold standard vs predicted segmentation
        segmentation_and_groundtruth_plot(binary_prediction_mask, ground_truth, subject_id, structure_name,
                                          plane, output_img_dir, annotator=annotator)

        if run_mode == SurfaceDistanceRunType.IOV:
            overall_gold_standard += ground_truth

        # Calculate and plot surface distance
        sds_full = sd_util.calculate_surface_distances(ground_truth, binary_prediction_mask, list(voxel_spacing))
        surface_distance_ground_truth_plot(ct, ground_truth, sds_full, subject_id, structure_name, plane, output_img_dir,
                                           annotator=annotator)

        if annotator is not None:
            sds_for_annotator[annotator] += sds_full

    # Plot all structures SDs for each annotator
    if run_mode == SurfaceDistanceRunType.IOV and subject_id is not None:
        for annotator, sds in sds_for_annotator.items():
            num_classes = int(np.amax(np.unique(overall_gold_standard)))
            binarised_gold_standard = multi_label_array_to_binary(overall_gold_standard, num_classes)[1:].sum(axis=0)
            surface_distance_ground_truth_plot(ct, binarised_gold_standard, sds, subject_id, 'All', plane, output_img_dir,
                                               annotator=annotator)


if __name__ == "__main__":
    main()

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import copy
import logging
import os
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from InnerEye.Azure.azure_util import PARENT_RUN_CONTEXT
from InnerEye.Common.common_util import METRICS_AGGREGATES_FILE, ModelProcessing, SUBJECT_METRICS_FILE_NAME, \
    get_epoch_results_path, is_linux, logging_section
from InnerEye.Common.fixed_paths import DEFAULT_RESULT_IMAGE_NAME
from InnerEye.Common.metrics_constants import MetricType, MetricsFileColumns
from InnerEye.ML import metrics, plotting
from InnerEye.ML.common import ModelExecutionMode, STORED_CSV_FILE_NAMES
from InnerEye.ML.config import DATASET_ID_FILE, GROUND_TRUTH_IDS_FILE, IMAGE_CHANNEL_IDS_FILE, SegmentationModelBase
from InnerEye.ML.dataset.full_image_dataset import FullImageDataset
from InnerEye.ML.dataset.sample import PatientMetadata, Sample
from InnerEye.ML.metrics import InferenceMetrics, InferenceMetricsForClassification, InferenceMetricsForSegmentation, \
    compute_scalar_metrics
from InnerEye.ML.metrics_dict import DataframeLogger, MetricsDict, ScalarMetricsDict, SequenceMetricsDict
from InnerEye.ML.model_config_base import ModelConfigBase
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import FullImageInferencePipelineBase, InferencePipeline, InferencePipelineBase
from InnerEye.ML.pipelines.scalar_inference import ScalarEnsemblePipeline, ScalarInferencePipeline, \
    ScalarInferencePipelineBase
from InnerEye.ML.reports.segmentation_report import boxplot_per_structure
from InnerEye.ML.scalar_config import ScalarModelBase
from InnerEye.ML.sequence_config import SequenceModelBase
from InnerEye.ML.utils import io_util, ml_util
from InnerEye.ML.utils.checkpoint_handling import CheckpointHandler
from InnerEye.ML.utils.image_util import binaries_from_multi_label_array
from InnerEye.ML.utils.io_util import ImageHeader, MedicalImageFileType, load_nifti_image, save_lines_to_file
from InnerEye.ML.utils.metrics_util import MetricsPerPatientWriter

BOXPLOT_FILE = "metrics_boxplot.png"
THUMBNAILS_FOLDER = "thumbnails"


def model_test(config: ModelConfigBase,
               data_split: ModelExecutionMode,
               checkpoint_handler: CheckpointHandler,
               model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> Optional[InferenceMetrics]:
    """
    Runs model inference on segmentation or classification models, using a given dataset (that could be training,
    test or validation set). The inference results and metrics will be stored and logged in a way that may
    differ for model categories (classification, segmentation).
    :param config: The configuration of the model
    :param data_split: Indicates which of the 3 sets (training, test, or validation) is being processed.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    :param model_proc: whether we are testing an ensemble or single model; this affects where results are written.
    :return: The metrics that the model achieved on the given data set, or None if the data set is empty.
    """
    if len(config.get_dataset_splits()[data_split]) == 0:
        logging.info(f"Skipping inference on empty data split {data_split}")
        return None
    if config.avoid_process_spawn_in_data_loaders and is_linux():
        logging.warning("Not performing any inference because avoid_process_spawn_in_data_loaders is set "
                        "and additional data loaders are likely to block.")
        return None
    with logging_section(f"Running {model_proc.value} model on {data_split.name.lower()} set"):
        if isinstance(config, SegmentationModelBase):
            return segmentation_model_test(config, data_split, checkpoint_handler, model_proc)
        if isinstance(config, ScalarModelBase):
            return classification_model_test(config, data_split, checkpoint_handler, model_proc)
    raise ValueError(f"There is no testing code for models of type {type(config)}")


def segmentation_model_test(config: SegmentationModelBase,
                            data_split: ModelExecutionMode,
                            checkpoint_handler: CheckpointHandler,
                            model_proc: ModelProcessing = ModelProcessing.DEFAULT) -> InferenceMetricsForSegmentation:
    """
    The main testing loop for segmentation models.
    It loads the model and datasets, then proceeds to test the model for all requested checkpoints.
    :param config: The arguments object which has a valid random seed attribute.
    :param data_split: Indicates which of the 3 sets (training, test, or validation) is being processed.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    :param model_proc: whether we are testing an ensemble or single model
    :return: InferenceMetric object that contains metrics related for all of the checkpoint epochs.
    """
    checkpoints_to_test = checkpoint_handler.get_checkpoints_to_test()

    if not checkpoints_to_test:
        raise ValueError("There were no checkpoints available for model testing.")

    epoch_results_folder = config.outputs_folder / get_epoch_results_path(data_split, model_proc)
    # save the datasets.csv used
    config.write_dataset_files(root=epoch_results_folder)
    epoch_and_split = f"{data_split.value} set"
    epoch_dice_per_image = segmentation_model_test_epoch(config=copy.deepcopy(config),
                                                         data_split=data_split,
                                                         checkpoint_paths=checkpoints_to_test,
                                                         results_folder=epoch_results_folder,
                                                         epoch_and_split=epoch_and_split)
    if epoch_dice_per_image is None:
        raise ValueError("There was no single checkpoint file available for model testing.")
    else:
        epoch_average_dice: float = np.mean(epoch_dice_per_image) if len(epoch_dice_per_image) > 0 else 0
        result = epoch_average_dice
        logging.info(f"Mean Dice: {epoch_average_dice:4f}")
        if model_proc == ModelProcessing.ENSEMBLE_CREATION:
            # For the upload, we want the path without the "OTHER_RUNS/ENSEMBLE" prefix.
            name = str(get_epoch_results_path(data_split, ModelProcessing.DEFAULT))
            PARENT_RUN_CONTEXT.upload_folder(name=name, path=str(epoch_results_folder))
    return InferenceMetricsForSegmentation(data_split=data_split, metrics=result)


def segmentation_model_test_epoch(config: SegmentationModelBase,
                                  data_split: ModelExecutionMode,
                                  checkpoint_paths: List[Path],
                                  results_folder: Path,
                                  epoch_and_split: str) -> Optional[List[float]]:
    """
    The main testing loop for a given epoch. It loads the model and datasets, then proceeds to test the model.
    Returns a list with an entry for each image in the dataset. The entry is the average Dice score,
    where the average is taken across all non-background structures in the image.
    :param checkpoint_paths: Checkpoint paths to run inference on.
    :param config: The arguments which specify all required information.
    :param data_split: Is the model evaluated on train, test, or validation set?
    :param results_folder: The folder where to store the results
    :param epoch_and_split: A string that should uniquely identify the epoch and the data split (train/val/test).
    :raises TypeError: If the arguments are of the wrong type.
    :raises ValueError: When there are issues loading the model.
    :return A list with the mean dice score (across all structures apart from background) for each image.
    """
    ml_util.set_random_seed(config.get_effective_random_seed(), "Model testing")
    results_folder.mkdir(exist_ok=True)

    test_dataframe = config.get_dataset_splits()[data_split]
    test_csv_path = results_folder / STORED_CSV_FILE_NAMES[data_split]
    test_dataframe.to_csv(path_or_buf=test_csv_path, index=False)
    logging.info("Results directory: {}".format(results_folder))
    logging.info(f"Starting evaluation of model {config.model_name} on {epoch_and_split}")

    # Write the dataset id and ground truth ids into the results folder
    store_run_information(results_folder, config.azure_dataset_id, config.ground_truth_ids, config.image_channels)

    ds = config.get_torch_dataset_for_inference(data_split)

    inference_pipeline = create_inference_pipeline(config=config, checkpoint_paths=checkpoint_paths)

    if inference_pipeline is None:
        # This will happen if there is no checkpoint for the given epoch, in either the recovered run (if any) or
        # the current one.
        return None

    # for mypy
    assert isinstance(inference_pipeline, FullImageInferencePipelineBase)

    # Deploy the trained model on a set of images and store output arrays.
    for sample_index, sample in enumerate(ds, 1):
        logging.info(f"Predicting for image {sample_index} of {len(ds)}...")
        sample = Sample.from_dict(sample=sample)
        inference_result = inference_pipeline.predict_and_post_process_whole_image(
            image_channels=sample.image,
            mask=sample.mask,
            patient_id=sample.patient_id,
            voxel_spacing_mm=sample.metadata.image_header.spacing
        )
        store_inference_results(inference_result=inference_result,
                                config=config,
                                results_folder=results_folder,
                                image_header=sample.metadata.image_header)

    # Evaluate model generated segmentation maps.
    num_workers = min(cpu_count(), len(ds))
    with Pool(processes=num_workers) as pool:
        pool_outputs = pool.map(
            partial(evaluate_model_predictions,
                    config=config,
                    dataset=ds,
                    results_folder=results_folder),
            range(len(ds)))

    average_dice = list()
    metrics_writer = MetricsPerPatientWriter()
    for (patient_metadata, metrics_for_patient) in pool_outputs:
        # Add the Dice score for the foreground classes, stored in the default hue
        metrics.add_average_foreground_dice(metrics_for_patient)
        average_dice.append(metrics_for_patient.get_single_metric(MetricType.DICE))
        # Structure names does not include the background class (index 0)
        for structure_name in config.ground_truth_ids:
            dice_for_struct = metrics_for_patient.get_single_metric(MetricType.DICE, hue=structure_name)
            hd_for_struct = metrics_for_patient.get_single_metric(MetricType.HAUSDORFF_mm, hue=structure_name)
            md_for_struct = metrics_for_patient.get_single_metric(MetricType.MEAN_SURFACE_DIST_mm, hue=structure_name)
            metrics_writer.add(patient=str(patient_metadata.patient_id),
                               structure=structure_name,
                               dice=dice_for_struct,
                               hausdorff_distance_mm=hd_for_struct,
                               mean_distance_mm=md_for_struct)

    metrics_writer.to_csv(results_folder / SUBJECT_METRICS_FILE_NAME)
    metrics_writer.save_aggregates_to_csv(results_folder / METRICS_AGGREGATES_FILE)
    if config.is_plotting_enabled:
        plt.figure()
        boxplot_per_structure(metrics_writer.to_data_frame(),
                              column_name=MetricsFileColumns.DiceNumeric.value,
                              title=f"Dice score for {epoch_and_split}")
        # The box plot file will be written to the output directory. AzureML will pick that up, and display
        # on the run overview page, without having to log to the run context.
        plotting.resize_and_save(5, 4, results_folder / BOXPLOT_FILE)
        plt.close()
    logging.info(f"Finished evaluation of model {config.model_name} on {epoch_and_split}")

    return average_dice


def evaluate_model_predictions(process_id: int,
                               config: SegmentationModelBase,
                               dataset: FullImageDataset,
                               results_folder: Path) -> Tuple[PatientMetadata, MetricsDict]:
    """
    Evaluates model segmentation predictions, dice scores and surface distances are computed.
    Generated contours are plotted and saved in results folder.
    The function is intended to be used in parallel for loop to process each image in parallel.
    :param process_id: Identifier for the process calling the function
    :param config: Segmentation model config object
    :param dataset: Dataset object, it is used to load intensity image, labels, and patient metadata.
    :param results_folder: Path to results folder
    :returns [PatientMetadata, list[list]]: Patient metadata and list of computed metrics for each image.
    """
    sample = dataset.get_samples_at_index(index=process_id)[0]
    logging.info(f"Evaluating predictions for patient {sample.patient_id}")
    patient_results_folder = get_patient_results_folder(results_folder, sample.patient_id)
    segmentation = load_nifti_image(patient_results_folder / DEFAULT_RESULT_IMAGE_NAME).image
    metrics_per_class = metrics.calculate_metrics_per_class(segmentation,
                                                            sample.labels,
                                                            ground_truth_ids=config.ground_truth_ids,
                                                            voxel_spacing=sample.image_spacing,
                                                            patient_id=sample.patient_id)
    thumbnails_folder = results_folder / THUMBNAILS_FOLDER
    thumbnails_folder.mkdir(exist_ok=True)
    plotting.plot_contours_for_all_classes(sample,
                                           segmentation=segmentation,
                                           foreground_class_names=config.ground_truth_ids,
                                           result_folder=thumbnails_folder,
                                           image_range=config.output_range)
    return sample.metadata, metrics_per_class


def get_patient_results_folder(results_folder: Path, patient_id: int) -> Path:
    """
    Gets a folder name that will contain all results for a given patient, like root/017 for patient 17.
    The folder name is constructed such that string sorting gives numeric sorting.
    :param results_folder: The root folder in which the per-patient results should sit.
    :param patient_id: The numeric ID of the patient.
    :return: A path like "root/017"
    """
    return results_folder / f"{int(patient_id):03d}"


def store_inference_results(inference_result: InferencePipeline.Result,
                            config: SegmentationModelBase,
                            results_folder: Path,
                            image_header: ImageHeader) -> List[Path]:
    """
    Store the segmentation, posteriors, and binary predictions into Nifti files.
    :param inference_result: The inference result for a given patient_id and epoch. Posteriors must be in
    (Classes x Z x Y x X) shape, segmentation in (Z, Y, X)
    :param config: The test configurations.
    :param results_folder: The folder where the prediction should be stored.
    :param image_header: The image header that was used in the input image.
    """

    def create_file_path(_results_folder: Path, _file_name: str) -> Path:
        """
        Create filename with Nifti extension
        :param _results_folder: The results folder
        :param _file_name: The name of the file
        :return: A full path to the results folder for the file
        """
        file_path = _file_name + MedicalImageFileType.NIFTI_COMPRESSED_GZ.value
        return _results_folder / file_path

    # create the directory for the given patient inside the results dir
    patient_results_folder = get_patient_results_folder(results_folder, inference_result.patient_id)
    patient_results_folder.mkdir(exist_ok=True, parents=True)

    # write the segmentations to disk
    image_paths = [io_util.store_as_ubyte_nifti(
        image=inference_result.segmentation,
        header=image_header,
        file_name=create_file_path(patient_results_folder, "segmentation"))]

    class_names_and_indices = config.class_and_index_with_background().items()
    binaries = binaries_from_multi_label_array(inference_result.segmentation, config.number_of_classes)
    # rescale posteriors if required and save them
    for (class_name, index), binary in zip(class_names_and_indices, binaries):
        posterior = inference_result.posteriors[index, ...]

        # save the posterior map
        file_name = "posterior_{}".format(class_name)
        image_path = io_util.store_posteriors_as_nifti(
            image=posterior,
            header=image_header,
            file_name=str(create_file_path(patient_results_folder, file_name)))
        image_paths.append(image_path)

        # save the binary mask
        image_path = io_util.store_binary_mask_as_nifti(
            image=binary,
            header=image_header,
            file_name=str(create_file_path(patient_results_folder, class_name)))
        image_paths.append(image_path)

    # rescale and store uncertainty map as nifti
    image_path = io_util.store_posteriors_as_nifti(
        image=inference_result.uncertainty,
        header=image_header,
        file_name=str(create_file_path(patient_results_folder, "uncertainty")))
    image_paths.append(image_path)
    return image_paths


def store_run_information(results_folder: Path,
                          dataset_id: Optional[str],
                          ground_truth_ids: List[str],
                          image_channels: List[str]) -> None:
    """
    Store dataset id and ground truth ids into files in the results folder.
    :param image_channels: The names of the image channels that the model consumes.
    :param results_folder: The folder where the files should be stored.
    :param dataset_id: The dataset id
    :param ground_truth_ids: The list of ground truth ids
    """
    # Recovery runs will download the previous job's output folder, it could contain these files.
    # Save these files for each epoch to keep the folders self-contained
    # Save the dataset id to a file
    save_lines_to_file(results_folder / DATASET_ID_FILE, [dataset_id or ""])
    # Save ground truth ids in a file
    save_lines_to_file(results_folder / GROUND_TRUTH_IDS_FILE, ground_truth_ids)
    # Save channel ids in a file
    save_lines_to_file(results_folder / IMAGE_CHANNEL_IDS_FILE, image_channels)


def create_inference_pipeline(config: ModelConfigBase,
                              checkpoint_paths: List[Path]) -> Optional[InferencePipelineBase]:
    """
    If multiple checkpoints are found in run_recovery then create EnsemblePipeline otherwise InferencePipeline.
    If no checkpoint files exist in the run recovery or current run checkpoint folder, None will be returned.
    :param config: Model related configs.
    :param epoch: The epoch for which to create pipeline for.
    :param run_recovery: RunRecovery data if applicable
    :return: FullImageInferencePipelineBase or ScalarInferencePipelineBase
    """
    if not checkpoint_paths:
        return None

    if len(checkpoint_paths) > 1:
        if config.is_segmentation_model:
            assert isinstance(config, SegmentationModelBase)
            return EnsemblePipeline.create_from_checkpoints(path_to_checkpoints=checkpoint_paths, model_config=config)
        elif config.is_scalar_model:
            assert isinstance(config, ScalarModelBase)
            return ScalarEnsemblePipeline.create_from_checkpoint(paths_to_checkpoint=checkpoint_paths, config=config)
        else:
            raise NotImplementedError("Cannot create inference pipeline for unknown model type")
    if len(checkpoint_paths) == 1:
        if config.is_segmentation_model:
            assert isinstance(config, SegmentationModelBase)
            return InferencePipeline.create_from_checkpoint(path_to_checkpoint=checkpoint_paths[0],
                                                            model_config=config)
        elif config.is_scalar_model:
            assert isinstance(config, ScalarModelBase)
            return ScalarInferencePipeline.create_from_checkpoint(path_to_checkpoint=checkpoint_paths[0],
                                                                  config=config)
        else:
            raise NotImplementedError("Cannot create ensemble pipeline for unknown model type")
    return None


def create_metrics_dict_for_scalar_models(config: ScalarModelBase) -> \
        Union[ScalarMetricsDict, SequenceMetricsDict]:
    """
    Create an instance of either a ScalarMetricsDict or SequenceMetricsDict, depending on whether the given
     configuration is a sequence model configuration or not.
    """
    if isinstance(config, SequenceModelBase):
        return SequenceMetricsDict.create(is_classification_model=config.is_classification_model,
                                          sequence_target_positions=config.sequence_target_positions)
    else:
        return ScalarMetricsDict(hues=config.target_names,
                                 is_classification_metrics=config.is_classification_model)


def classification_model_test(config: ScalarModelBase,
                              data_split: ModelExecutionMode,
                              checkpoint_handler: CheckpointHandler,
                              model_proc: ModelProcessing) -> InferenceMetricsForClassification:
    """
    The main testing loop for classification models. It runs a loop over all epochs for which testing should be done.
    It loads the model and datasets, then proceeds to test the model for all requested checkpoints.
    :param config: The model configuration.
    :param data_split: The name of the folder to store the results inside each epoch folder in the outputs_dir,
                       used mainly in model evaluation using different dataset splits.
    :param checkpoint_handler: Checkpoint handler object to find checkpoint paths for model initialization
    :param model_proc: whether we are testing an ensemble or single model
    :return: InferenceMetricsForClassification object that contains metrics related for all of the checkpoint epochs.
    """
    posthoc_label_transform = config.get_posthoc_label_transform()

    def test_epoch(checkpoint_paths: List[Path]) -> Optional[MetricsDict]:
        pipeline = create_inference_pipeline(config=config,
                                             checkpoint_paths=checkpoint_paths)

        if pipeline is None:
            return None

        # for mypy
        assert isinstance(pipeline, ScalarInferencePipelineBase)

        ml_util.set_random_seed(config.get_effective_random_seed(), "Model Testing")
        ds = config.get_torch_dataset_for_inference(data_split).as_data_loader(
            shuffle=False,
            batch_size=1,
            num_dataload_workers=0
        )

        logging.info(f"Starting to evaluate model on {data_split.value} set.")
        metrics_dict = create_metrics_dict_for_scalar_models(config)
        for sample in ds:
            result = pipeline.predict(sample)
            model_output = result.posteriors
            label = result.labels.to(device=model_output.device)
            label = posthoc_label_transform(label)
            sample_id = result.subject_ids[0]
            compute_scalar_metrics(metrics_dict,
                                   subject_ids=[sample_id],
                                   model_output=model_output,
                                   labels=label,
                                   loss_type=config.loss_type)
            logging.debug(f"Example {sample_id}: {metrics_dict.to_string()}")

        average = metrics_dict.average(across_hues=False)
        logging.info(average.to_string())

        return metrics_dict

    checkpoints_to_test = checkpoint_handler.get_checkpoints_to_test()

    if not checkpoints_to_test:
        raise ValueError("There were no checkpoints available for model testing.")

    result = test_epoch(checkpoint_paths=checkpoints_to_test)
    if result is None:
        raise ValueError("There was no single checkpoint file available for model testing.")
    else:
        if isinstance(result, ScalarMetricsDict):
            results_folder = config.outputs_folder / get_epoch_results_path(data_split, model_proc)
            csv_file = results_folder / SUBJECT_METRICS_FILE_NAME

            logging.info(f"Writing {data_split.value} metrics to file {str(csv_file)}")

            # If we are running inference after a training run, the validation set metrics may have been written
            # during train time. If this is not the case, or we are running on the test set, create the metrics
            # file.
            if not csv_file.exists():
                os.makedirs(str(results_folder), exist_ok=False)
                df_logger = DataframeLogger(csv_file)

                # cross validation split index not relevant during test time
                result.store_metrics_per_subject(df_logger=df_logger,
                                                 mode=data_split)
                # write to disk
                df_logger.flush()

    return InferenceMetricsForClassification(metrics=result)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from io import StringIO
from typing import Any, List, Optional, Tuple
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch

from InnerEye.Common import common_util
from InnerEye.Common.common_util import METRICS_FILE_NAME, ModelExecutionMode, logging_to_stdout
from InnerEye.Common.metrics_dict import MetricType, SequenceMetricsDict
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.ML.dataset.sequence_dataset import SequenceDataset
from InnerEye.ML.deep_learning_config import TemperatureScalingConfig
from InnerEye.ML.model_config_base import ModelTransformsPerExecutionMode
from InnerEye.ML.model_training import model_train
from InnerEye.ML.model_training_steps import get_scalar_model_inputs_and_labels
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImageEncoder, ImagingFeatureType
from InnerEye.ML.models.architectures.sequential.rnn_classifier import RNNClassifier, RNNClassifierWithEncoder
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.scalar_config import ScalarLoss
from InnerEye.ML.sequence_config import SEQUENCE_LENGTH_FILE, SEQUENCE_LENGTH_STATS_FILE, \
    SEQUENCE_POSITION_HUE_NAME_PREFIX, SequenceModelBase
from InnerEye.ML.utils import ml_util
from InnerEye.ML.utils.augmentation import RandAugmentSlice, ScalarItemAugmentation
from InnerEye.ML.utils.dataset_util import CategoricalToOneHotEncoder
from InnerEye.ML.utils.io_util import ImageAndSegmentations
from InnerEye.ML.utils.metrics_constants import LoggingColumns
from InnerEye.ML.utils.model_util import ModelAndInfo, create_model_with_temperature_scaling
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.visualizers.grad_cam_hooks import VisualizationMaps
from Tests.ML.util import get_default_checkpoint_handler, get_default_azure_config
from Tests.fixed_paths_for_tests import full_ml_test_data_path

SCAN_SIZE = (6, 64, 60)


def prepare_sequences(num_sequences: int, sequence_length: int, batch_size: int) -> Tuple[List, List]:
    # Returns [batch][sequence, label]
    num_mini_batches = num_sequences // batch_size

    inputs = np.random.choice([0, 1], size=(num_sequences, sequence_length), p=[1. / 3, 2. / 3]).astype(np.float32)
    inputs = torch.tensor(inputs)
    labels = torch.sum(inputs, dim=1) > (sequence_length // 2)
    labels = labels.long()
    data = list()

    for batch_index in range(num_mini_batches):
        _input = inputs[batch_index * batch_size: (batch_index + 1) * batch_size]
        _label = labels[batch_index * batch_size: (batch_index + 1) * batch_size]
        data.append((_input, _label))

    return data[:num_mini_batches // 2], data[num_mini_batches // 2:]


class ToySequenceModel(SequenceModelBase):
    def __init__(self, use_combined_model: bool = False,
                 imaging_feature_type: ImagingFeatureType = ImagingFeatureType.Image,
                 combine_hidden_states: bool = False,
                 use_encoder_layer_norm: bool = False,
                 sequence_target_positions: Optional[List[int]] = None,
                 use_mean_teacher_model: bool = False,
                 **kwargs: Any) -> None:
        num_epochs = 3
        mean_teacher_alpha = 0.999 if use_mean_teacher_model else None
        sequence_target_positions = [2] if sequence_target_positions is None else sequence_target_positions
        image_column = "image" if use_combined_model else None
        categorical_feature_encoder = CategoricalToOneHotEncoder.create_from_dataframe(
            dataframe=_get_mock_sequence_dataset(), columns=["cat1"])
        super().__init__(
            local_dataset=full_ml_test_data_path("sequence_data_for_classification"),
            temperature_scaling_config=TemperatureScalingConfig(),
            label_value_column="label",
            numerical_columns=["numerical1", "numerical2"],
            categorical_columns=["cat1"],
            categorical_feature_encoder=categorical_feature_encoder,
            sequence_column="seqColumn",
            sequence_target_positions=sequence_target_positions,
            image_file_column=image_column,
            loss_type=ScalarLoss.WeightedCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            test_start_epoch=num_epochs,
            train_batch_size=3,
            l_rate=1e-1,
            load_segmentation=True,
            use_mixed_precision=True,
            label_smoothing_eps=0.05,
            drop_last_batch_in_training=True,
            mean_teacher_alpha=mean_teacher_alpha,
            **kwargs
        )
        self.use_combined_model = use_combined_model
        self.imaging_feature_type = imaging_feature_type
        self.combine_hidden_state = combine_hidden_states
        self.use_encoder_layer_norm = use_encoder_layer_norm

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
        )

    def get_image_sample_transforms(self) -> ModelTransformsPerExecutionMode:
        if self.use_combined_model:
            return ModelTransformsPerExecutionMode(
                train=ScalarItemAugmentation(
                    transform=RandAugmentSlice(use_joint_channel_transformation=False,
                                               is_transformation_for_segmentation_maps=True)))
        else:
            return ModelTransformsPerExecutionMode()

    def create_model(self) -> RNNClassifier:
        if self.use_combined_model:
            image_encoder: Optional[ImageEncoder] = \
                ImageEncoder(num_image_channels=1,
                             imaging_feature_type=self.imaging_feature_type,
                             num_non_image_features=self.get_total_number_of_non_imaging_features(),
                             stride_size_per_encoding_block=(1, 2, 2),
                             initial_feature_channels=4,
                             num_encoder_blocks=3,
                             )
            assert image_encoder is not None  # for mypy
            input_dims = image_encoder.final_num_feature_channels
        else:
            image_encoder = None
            input_dims = self.get_total_number_of_non_imaging_features()

        ref_indices = [0, 1] if self.combine_hidden_state else None

        return RNNClassifierWithEncoder(input_dim=input_dims,
                                        hidden_dim=3,
                                        output_dim=1,
                                        num_rnn_layers=1,
                                        rnn_dropout=0.0,
                                        ref_indices=ref_indices,
                                        image_encoder=image_encoder,
                                        use_encoder_batch_norm=self.use_encoder_layer_norm,
                                        target_indices=self.get_target_indices())


def _get_mock_sequence_dataset(dataset_contents: Optional[str] = None) -> pd.DataFrame:
    # The dataset has "measurements" for 3 different positions 0, 1, and 2, with columns for numerical1 and numerical2.
    # Labels are attached to position 3 only.
    if dataset_contents is None:
        dataset_contents = """subject,numerical1,numerical2,cat1,seqColumn,label,image
2137.00005,362,71,A,0,0,scan1.npy
2137.00005,357,69,B,1,0,scan2.npy
2137.00005,355,64,C,2,0,scan3.npy
2137.00005,355,63,C,3,1,scan4.npy
2137.00125,348,64,A,0,0,scan1.npy
2137.00125,316,68,A,1,0,scan3.npy
2137.00125,349,68,A,2,0,scan2.npy
2137.00125,361,67,B,3,0,scan1.npy
2137.00125,350,68,B,4,0,scan1.npy
2627.00001,477,58,A,0,0,scan2.npy
2627.00001,220,59,A,1,0,scan2.npy
2627.00001,222,60,A,2,0,scan1.npy
2627.00001,217,65,A,5,1,scan3.npy
2627.12341,210,60,B,0,0,scan4.npy
2627.12341,217,61,B,1,0,scan1.npy
2627.12341,224,63,B,2,1,scan2.npy
3250.00005,344,76,C,0,0,scan2.npy
3250.00005,233,76,C,1,0,scan4.npy
3250.00005,212,84,C,2,0,scan3.npy
3250.00005,215,84,C,3,0,scan1.npy
3250.00005,215,82,C,4,0,scan1.npy
3250.12345,233,84,C,0,0,scan3.npy
3250.12345,218,84,C,1,0,scan3.npy
3250.12345,221,84,C,2,0,scan1.npy
3250.12345,238,84,C,3,0,scan1.npy
"""
    return pd.read_csv(StringIO(dataset_contents), dtype=str)


@pytest.mark.parametrize(["use_combined_model", "imaging_feature_type"],
                         [(False, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Segmentation),
                          (True, ImagingFeatureType.ImageAndSegmentation)])
@pytest.mark.parametrize("combine_hidden_state", (True, False))
@pytest.mark.parametrize("use_encoder_layer_norm", (True, False))
@pytest.mark.parametrize("use_mean_teacher_model", (True, False))
@pytest.mark.gpu
def test_rnn_classifier_via_config_1(use_combined_model: bool,
                                     imaging_feature_type: ImagingFeatureType,
                                     combine_hidden_state: bool,
                                     use_encoder_layer_norm: bool,
                                     use_mean_teacher_model: bool,
                                     test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can build a simple RNN model that only feeds off non-image features.
    This just tests the mechanics of training, but not if the model learned.
    """
    logging_to_stdout()
    config = ToySequenceModel(use_combined_model,
                              imaging_feature_type=imaging_feature_type,
                              combine_hidden_states=combine_hidden_state,
                              use_encoder_layer_norm=use_encoder_layer_norm,
                              use_mean_teacher_model=use_mean_teacher_model,
                              should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_mock_sequence_dataset()
    # Patch the load_images function that will be called once we access a dataset item
    image_and_seg = ImageAndSegmentations[np.ndarray](images=np.random.uniform(0, 1, SCAN_SIZE),
                                                      segmentations=np.random.randint(0, 2, SCAN_SIZE))
    with mock.patch('InnerEye.ML.utils.io_util.load_image_in_known_formats', return_value=image_and_seg):
        results = model_train(config, get_default_checkpoint_handler(model_config=config,
                                                                     project_root=test_output_dirs.root_dir))
        assert len(results.optimal_temperature_scale_values_per_checkpoint_epoch) \
               == config.get_total_number_of_save_epochs()


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.parametrize(["use_combined_model", "imaging_feature_type"],
                         [(False, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Segmentation),
                          (True, ImagingFeatureType.ImageAndSegmentation)])
def test_run_ml_with_sequence_model(use_combined_model: bool,
                                    imaging_feature_type: ImagingFeatureType,
                                    test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of sequence models, when it is started together via run_ml.
    """
    logging_to_stdout()
    config = ToySequenceModel(use_combined_model, imaging_feature_type,
                              should_validate=False, sequence_target_positions=[2, 10])
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_mock_sequence_dataset()
    config.num_epochs = 1
    config.max_batch_grad_cam = 1

    # make sure we are testing with at least one sequence position that will not exist
    # to ensure correct handling of sequences that do not contain all the expected target positions
    assert max(config.sequence_target_positions) > config.dataset_data_frame[config.sequence_column].astype(float).max()

    # Patch the load_images function that will be called once we access a dataset item
    image_and_seg = ImageAndSegmentations[np.ndarray](images=np.random.uniform(0, 1, SCAN_SIZE),
                                                      segmentations=np.random.randint(0, 2, SCAN_SIZE))
    with mock.patch('InnerEye.ML.utils.io_util.load_image_in_known_formats', return_value=image_and_seg):
        azure_config = get_default_azure_config()
        azure_config.train = True
        MLRunner(config, azure_config).run()


@pytest.mark.skipif(common_util.is_windows(), reason="Too slow on windows")
@pytest.mark.parametrize(["use_combined_model", "imaging_feature_type"],
                         [(False, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Image),
                          (True, ImagingFeatureType.Segmentation),
                          (True, ImagingFeatureType.ImageAndSegmentation)])
def test_visualization_with_sequence_model(use_combined_model: bool,
                                           imaging_feature_type: ImagingFeatureType,
                                           test_output_dirs: OutputFolderForTests) -> None:
    config = ToySequenceModel(use_combined_model, imaging_feature_type, should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_mock_sequence_dataset()
    config.num_epochs = 1

    model_and_info = ModelAndInfo(config=config, model_execution_mode=ModelExecutionMode.TEST,
                                  checkpoint_path=None)
    model_loaded = model_and_info.try_create_model_load_from_checkpoint_and_adjust()
    assert model_loaded

    model = model_and_info.model

    dataloader = SequenceDataset(config,
                                 data_frame=config.dataset_data_frame).as_data_loader(shuffle=False,
                                                                                      batch_size=2)
    # Patch the load_images function that will be called once we access a dataset item
    image_and_seg = ImageAndSegmentations[np.ndarray](images=np.random.uniform(0, 1, SCAN_SIZE),
                                                      segmentations=np.random.randint(0, 2, SCAN_SIZE))
    with mock.patch('InnerEye.ML.utils.io_util.load_image_in_known_formats', return_value=image_and_seg):
        batch = next(iter(dataloader))
        model_inputs_and_labels = get_scalar_model_inputs_and_labels(config, model, batch)  # type: ignore
    number_sequences = model_inputs_and_labels.model_inputs[0].shape[1]
    number_subjects = len(model_inputs_and_labels.subject_ids)
    visualizer = VisualizationMaps(model, config)
    guided_grad_cams, grad_cams, pseudo_cam_non_img, probas = visualizer.generate(
        model_inputs_and_labels.model_inputs)
    if use_combined_model:
        if imaging_feature_type == ImagingFeatureType.ImageAndSegmentation:
            assert guided_grad_cams.shape[:2] == (number_subjects, number_sequences * 2)
            assert grad_cams.shape[:2] == (number_subjects, number_sequences * 2)
        else:
            assert guided_grad_cams.shape[:2] == (number_subjects, number_sequences)
            assert grad_cams.shape[:2] == (number_subjects, number_sequences)
    else:
        assert guided_grad_cams is None
        assert grad_cams is None
        assert pseudo_cam_non_img.shape[:2] == (number_subjects, number_sequences)
        assert probas.shape[0] == number_subjects
    non_image_features = config.numerical_columns + config.categorical_columns
    non_imaging_plot_labels = visualizer._get_non_imaging_plot_labels(model_inputs_and_labels.data_item,
                                                                      non_image_features,
                                                                      index=0,
                                                                      target_position=3)
    assert non_imaging_plot_labels == ['numerical1_0',
                                       'numerical2_0',
                                       'cat1_0',
                                       'numerical1_1',
                                       'numerical2_1',
                                       'cat1_1',
                                       'numerical1_2',
                                       'numerical2_2',
                                       'cat1_2',
                                       'numerical1_3',
                                       'numerical2_3',
                                       'cat1_3']


class ToySequenceModel2(SequenceModelBase):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            temperature_scaling_config=TemperatureScalingConfig(),
            local_dataset=full_ml_test_data_path("sequence_data_for_classification"),
            label_value_column="label",
            numerical_columns=["feature"],
            sequence_column="index",
            sequence_target_positions=[2],
            loss_type=ScalarLoss.BinaryCrossEntropyWithLogits,
            num_epochs=20,
            num_dataload_workers=0,
            train_batch_size=40,
            l_rate=1e-2,
            drop_last_batch_in_training=True,
            **kwargs
        )

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.8,
            proportion_test=0.1,
            proportion_val=0.1,
        )

    def create_model(self) -> Any:
        return RNNClassifier(input_dim=self.get_total_number_of_non_imaging_features(),
                             hidden_dim=12,
                             output_dim=1,
                             num_rnn_layers=1,
                             rnn_dropout=0.25,
                             use_layer_norm=False,
                             target_indices=self.get_target_indices())


# Only test the non-combined model because otherwise the build takes too much time.
@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
@pytest.mark.gpu
def test_rnn_classifier_via_config_2(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test if we can build an RNN classifier that learns sequences, of the same kind as in
    test_rnn_classifier_toy_problem, but built via the config.
    """
    expected_max_train_loss = 0.71
    expected_max_val_loss = 0.71
    num_sequences = 100
    ml_util.set_random_seed(123)
    dataset_contents = "subject,index,feature,label\n"
    for subject in range(num_sequences):
        # Sequences have variable length
        sequence_length = np.random.choice([9, 10, 11, 12])
        # Each sequence is a series of 0 and 1
        inputs = np.random.choice([0, 1], size=(sequence_length,), p=[1. / 3, 2. / 3])
        label = np.sum(inputs) > (sequence_length // 2)
        for i, value in enumerate(inputs.tolist()):
            dataset_contents += f"S{subject},{i},{value},{label}\n"
    logging_to_stdout()
    config = ToySequenceModel2(should_validate=False)
    config.num_epochs = 2
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_mock_sequence_dataset(dataset_contents)
    results = model_train(config, get_default_checkpoint_handler(model_config=config,
                                                                 project_root=test_output_dirs.root_dir))

    actual_train_loss = results.train_results_per_epoch[-1].values()[MetricType.LOSS.value][0]
    actual_val_loss = results.val_results_per_epoch[-1].values()[MetricType.LOSS.value][0]
    print(f"Training loss after {config.num_epochs} epochs: {actual_train_loss}")
    print(f"Validation loss after {config.num_epochs} epochs: {actual_val_loss}")
    assert actual_train_loss <= expected_max_train_loss, "Training loss too high"
    assert actual_val_loss <= expected_max_val_loss, "Validation loss too high"
    assert len(results.optimal_temperature_scale_values_per_checkpoint_epoch) \
           == config.get_total_number_of_save_epochs()
    assert np.allclose(results.optimal_temperature_scale_values_per_checkpoint_epoch, [0.97], rtol=0.1)


class ToyMultiLabelSequenceModel(SequenceModelBase):
    def __init__(self, **kwargs: Any) -> None:
        num_epochs = 3
        super().__init__(
            temperature_scaling_config=TemperatureScalingConfig(),
            label_value_column="Label",
            numerical_columns=["NUM1", "NUM2"],
            sequence_column="Position",
            sequence_target_positions=[1, 2, 3],
            loss_type=ScalarLoss.WeightedCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            test_start_epoch=num_epochs,
            train_batch_size=3,
            l_rate=1e-1,
            label_smoothing_eps=0.05,
            categorical_columns=["CAT1"],
            **kwargs
        )

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
        )

    def create_model(self) -> Any:
        return RNNClassifier(input_dim=self.get_total_number_of_non_imaging_features(),
                             hidden_dim=3,
                             output_dim=1,
                             num_rnn_layers=2,
                             rnn_dropout=0.0,
                             target_indices=self.get_target_indices())


@pytest.mark.skipif(common_util.is_windows(), reason="Has issues on windows build")
def test_run_ml_with_multi_label_sequence_model(test_output_dirs: OutputFolderForTests) -> None:
    """
    Test training and testing of sequence models that predicts at multiple time points,
    when it is started via run_ml.
    """
    logging_to_stdout()
    config = ToyMultiLabelSequenceModel(should_validate=False)
    assert config.get_target_indices() == [1, 2, 3]
    expected_prediction_targets = [f"{SEQUENCE_POSITION_HUE_NAME_PREFIX} {x}"
                                   for x in ["01", "02", "03"]]
    _target_indices = config.get_target_indices()
    assert _target_indices is not None
    assert len(_target_indices) == len(expected_prediction_targets)
    metrics_dict = SequenceMetricsDict.create_from_config(config)
    assert metrics_dict.get_hue_names(include_default=False) == expected_prediction_targets
    config.set_output_to(test_output_dirs.root_dir)
    # Create a fake dataset directory to make config validation pass
    config.local_dataset = test_output_dirs.root_dir
    config.dataset_data_frame = _get_multi_label_sequence_dataframe()
    config.pre_process_dataset_dataframe()
    config.num_epochs = 1
    config.max_batch_grad_cam = 1
    azure_config = get_default_azure_config()
    azure_config.train = True
    MLRunner(config, azure_config).run()
    # The metrics file should have one entry per epoch per subject per prediction target,
    # for all the 3 prediction targets.
    metrics_file = config.outputs_folder / "Train" / METRICS_FILE_NAME
    assert metrics_file.exists()
    metrics = pd.read_csv(metrics_file)
    assert LoggingColumns.Patient.value in metrics
    assert LoggingColumns.Epoch.value in metrics
    assert LoggingColumns.Hue.value in metrics
    assert metrics[LoggingColumns.Hue.value].unique().tolist() == expected_prediction_targets
    group_by_subject = metrics.groupby(by=[LoggingColumns.Patient.value,
                                           LoggingColumns.Epoch.value])
    expected_prediction_target_lengths = [3, 2, 3, 3]
    for i, x in enumerate(group_by_subject):
        assert len(x[1]) == expected_prediction_target_lengths[i]
    group_by_subject_and_target = metrics.groupby(by=[LoggingColumns.Patient.value,
                                                      LoggingColumns.Epoch.value,
                                                      LoggingColumns.Hue.value])
    for _, group in group_by_subject_and_target:
        assert len(group) == 1


@pytest.mark.parametrize("combine_hidden_states", [True, False])
def test_pad_gru_output(combine_hidden_states: bool) -> None:
    """
    Test to make sure if model output does not cover the target indices then it is padded
    """
    config = ToySequenceModel(
        sequence_target_positions=[5, 7],
        combine_hidden_states=combine_hidden_states,
        should_validate=False
    )
    model: RNNClassifier = config.create_model()
    # base case where no padding is required
    test_input = torch.rand(max(config.get_target_indices()) + 1, 1)
    padded = model.pad_gru_output(test_input)
    assert torch.equal(test_input, padded)
    # case when padding is required
    test_input = torch.rand(min(config.get_target_indices()) - 1, 1)
    expected = torch.cat([test_input, test_input.new_full((4, 1), fill_value=0)], dim=0)
    padded = model.pad_gru_output(test_input)
    assert torch.allclose(expected, padded)


def test_visualization_for_different_target_weeks(test_output_dirs: OutputFolderForTests) -> None:
    """
    Tests that the visualizations are differentiated depending on the target week
    for which we visualize it.
    """
    config = ToyMultiLabelSequenceModel(should_validate=False)
    config.set_output_to(test_output_dirs.root_dir)
    config.dataset_data_frame = _get_multi_label_sequence_dataframe()
    config.pre_process_dataset_dataframe()
    model = create_model_with_temperature_scaling(config)
    dataloader = SequenceDataset(config,
                                 data_frame=config.dataset_data_frame).as_data_loader(shuffle=False,
                                                                                      batch_size=2)
    batch = next(iter(dataloader))
    model_inputs_and_labels = get_scalar_model_inputs_and_labels(config, model, batch)  # type: ignore

    visualizer = VisualizationMaps(model, config)
    # Pseudo-grad cam explaining the prediction at target sequence 2
    _, _, pseudo_cam_non_img_3, probas_3 = visualizer.generate(model_inputs_and_labels.model_inputs,
                                                               target_position=2,
                                                               target_label_index=2)
    # Pseudo-grad cam explaining the prediction at target sequence 0
    _, _, pseudo_cam_non_img_1, probas_1 = visualizer.generate(model_inputs_and_labels.model_inputs,
                                                               target_position=0,
                                                               target_label_index=0)
    assert pseudo_cam_non_img_1.shape[1] == 1
    assert pseudo_cam_non_img_3.shape[1] == 3
    # Both visualizations should not be equal
    assert np.any(pseudo_cam_non_img_1 != pseudo_cam_non_img_3)
    assert np.any(probas_3 != probas_1)


def _get_multi_label_sequence_dataframe() -> pd.DataFrame:
    """
    Returns a mock dataset for multi label sequence model.
    """
    dataset_contents = """subject,NUM1,CAT1,NUM2,Position,Label
2137.00005,362,A,71,0,
2137.00005,357,B,69,1,0
2137.00005,355,C,64,2,0
2137.00005,355,C,63,3,1
2137.00125,348,A,64,0,0
2137.00125,316,A,68,1,1
2137.00125,349,B,68,2,0
2137.00125,361,B,67,3,1
2137.00125,350,B,68,4,0
2627.00001,477,C,58,0,0
2627.00001,220,C,59,1,0
2627.00001,222,A,60,2,0
2627.00001,217,A,65,5,1
2627.12341,210,B,60,0,0
2627.12341,217,B,61,1,0
2627.12341,224,B,63,2,1
3250.00005,344,B,76,0,0
3250.00005,233,A,76,1,0
3250.00005,212,A,84,2,0
3250.00005,215,A,84,3,0
3250.00005,215,A,82,4,0
3250.12345,233,A,84,0,1
3250.12345,218,A,84,1,0
3250.12345,221,B,84,2,0
3250.12345,238,B,84,3,0
"""
    return pd.read_csv(StringIO(dataset_contents), dtype=str)


def test_sequence_dataset_stats_hook(test_output_dirs: OutputFolderForTests) -> None:
    model = ToySequenceModel()
    model.set_output_to(test_output_dirs.root_dir)
    model.dataset_data_frame = _get_mock_sequence_dataset()
    model.create_and_set_torch_datasets()
    length_file = model.logs_folder / SEQUENCE_LENGTH_FILE
    assert length_file.is_file()
    assert length_file.read_text().splitlines() == [
        "cross_validation_split_index,data_split,subject,sequence_length",
        "-1,Train,2137.00005,4",
        "-1,Train,2627.12341,3",
        "-1,Train,3250.00005,5",
        "-1,Train,3250.12345,4",
        "-1,Test,2627.00001,3",
        "-1,Val,2137.00125,5"]
    stats_file = model.logs_folder / SEQUENCE_LENGTH_STATS_FILE
    assert stats_file.is_file()
    assert stats_file.read_text().splitlines() == [
        "           sequence_length                                          ",
        "                     count mean       std  min   25%  50%   75%  max",
        "data_split                                                          ",
        "Test                   1.0  3.0       NaN  3.0  3.00  3.0  3.00  3.0",
        "Train                  4.0  4.0  0.816497  3.0  3.75  4.0  4.25  5.0",
        "Val                    1.0  5.0       NaN  5.0  5.00  5.0  5.00  5.0"]

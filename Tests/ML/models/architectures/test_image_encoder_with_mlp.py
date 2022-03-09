#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import time
from io import StringIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import ColorJitter, RandomAffine

from InnerEye.Common.common_util import logging_to_stdout
from InnerEye.Common.output_directories import OutputFolderForTests
from InnerEye.Common.type_annotations import TupleInt3
from InnerEye.ML.augmentations.transform_pipeline import ImageTransformationPipeline
from InnerEye.ML.dataset.scalar_dataset import ScalarDataset, ScalarItemAugmentation
from InnerEye.ML.model_config_base import ModelTransformsPerExecutionMode
from InnerEye.ML.models.architectures.classification.image_encoder_with_mlp import ImageEncoderWithMlp, \
    ImagingFeatureType
from InnerEye.ML.run_ml import MLRunner
from InnerEye.ML.scalar_config import AggregationType, ScalarLoss, ScalarModelBase, get_non_image_features_dict
from InnerEye.ML.utils.image_util import HDF5_NUM_SEGMENTATION_CLASSES, segmentation_to_one_hot
from InnerEye.ML.utils.io_util import ImageAndSegmentations, NumpyFile
from InnerEye.ML.utils.ml_util import is_gpu_available, set_random_seed
from InnerEye.ML.utils.model_util import create_model_with_temperature_scaling
from InnerEye.ML.utils.split_dataset import DatasetSplits
from InnerEye.ML.visualizers.model_summary import ModelSummary
from Tests.ML.util import get_default_azure_config, model_train_unittest


class ImageEncoder(ScalarModelBase):
    def __init__(self,
                 encode_channels_jointly: bool,
                 imaging_feature_type: ImagingFeatureType = ImagingFeatureType.Image,
                 kernel_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 3, 3),
                 stride_size_per_encoding_block: Union[TupleInt3, List[TupleInt3]] = (1, 2, 2),
                 encoder_dimensionality_reduction_factor: float = 0.8,
                 aggregation_type: AggregationType = AggregationType.Average,
                 scan_size: Optional[TupleInt3] = None,
                 **kwargs: Any) -> None:
        num_epochs = 3
        super().__init__(
            image_channels=["week0", "week1"],
            image_file_column="path",
            label_channels=["week1"],
            label_value_column="label",
            loss_type=ScalarLoss.WeightedCrossEntropyWithLogits,
            num_epochs=num_epochs,
            num_dataload_workers=0,
            train_batch_size=16,
            l_rate=1e-1,
            use_mixed_precision=True,
            aggregation_type=aggregation_type,
            azure_dataset_id="test-dataset",
            **kwargs
        )
        self.encode_channels_jointly = encode_channels_jointly
        self.imaging_feature_type = imaging_feature_type
        self.load_segmentation = (imaging_feature_type == ImagingFeatureType.Segmentation
                                  or imaging_feature_type == ImagingFeatureType.ImageAndSegmentation)
        self.kernel_size_per_encoding_block = kernel_size_per_encoding_block
        self.stride_size_per_encoding_block = stride_size_per_encoding_block
        self.encoder_dimensionality_reduction_factor = encoder_dimensionality_reduction_factor
        self.size_input = scan_size
        # Trying to run DDP from the test suite hangs, hence restrict to single GPU.
        self.max_num_gpus = 1
        self.generate_report = False

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        return DatasetSplits.from_proportions(
            df=dataset_df,
            proportion_train=0.7,
            proportion_test=0.2,
            proportion_val=0.1,
        )

    def create_model(self) -> Any:
        return ImageEncoderWithMlp(
            encode_channels_jointly=self.encode_channels_jointly,
            num_image_channels=len(self.image_channels),
            initial_feature_channels=4,
            num_encoder_blocks=3,
            mlp_dropout=0.5,
            imaging_feature_type=self.imaging_feature_type,
            num_non_image_features=self.get_total_number_of_non_imaging_features(),
            kernel_size_per_encoding_block=self.kernel_size_per_encoding_block,
            stride_size_per_encoding_block=self.stride_size_per_encoding_block,
            encoder_dimensionality_reduction_factor=self.encoder_dimensionality_reduction_factor,
            aggregation_type=self.aggregation_type,
            scan_size=self.size_input
        )

    def get_post_loss_logits_normalization_function(self) -> Callable:
        return torch.nn.Sigmoid()

    def get_image_transform(self) -> ModelTransformsPerExecutionMode:
        """
        Get transforms to perform on image samples for each model execution mode.
        """
        if self.imaging_feature_type in [ImagingFeatureType.Image, ImagingFeatureType.ImageAndSegmentation]:
            return ModelTransformsPerExecutionMode(
                train=ImageTransformationPipeline(
                    transforms=[RandomAffine(10), ColorJitter(0.2)],
                    use_different_transformation_per_channel=True))
        return ModelTransformsPerExecutionMode()

    def get_segmentation_transform(self) -> ModelTransformsPerExecutionMode:
        if self.imaging_feature_type in [ImagingFeatureType.Segmentation, ImagingFeatureType.ImageAndSegmentation]:
            return ModelTransformsPerExecutionMode(
                train=ImageTransformationPipeline(
                    transforms=[RandomAffine(10), ColorJitter(0.2)],
                    use_different_transformation_per_channel=True))
        return ModelTransformsPerExecutionMode()


@pytest.mark.parametrize(["encode_channels_jointly", "use_non_imaging_features",
                          "reduction_factor", "expected_num_reduced_features",
                          "kernel_size_per_encoding_block", "stride_size_per_encoding_block",
                          "aggregation_type"],
                         [(True, True, 1, 8, None, None, AggregationType.Average),
                          (True, True, 1, 8, None, None, AggregationType.ZAdaptive3dAvg),
                          (True, True, 1, 8, None, None, AggregationType.GatedPooling),
                          (False, True, 1, 8, None, None, AggregationType.Average),
                          (False, True, 0.1, 1, None, None, AggregationType.Average),
                          (False, True, 0.5, 4, None, None, AggregationType.Average),
                          (False, False, 1, 0, None, None, AggregationType.Average),
                          (True, True, 1, 8, [(1, 1, 1), (1, 3, 3), (3, 3, 3)], [(1, 1, 1), (1, 2, 2), (2, 2, 2)],
                           AggregationType.Average),
                          ])
def test_image_encoder(test_output_dirs: OutputFolderForTests,
                       encode_channels_jointly: bool,
                       use_non_imaging_features: bool,
                       kernel_size_per_encoding_block: Optional[Union[TupleInt3, List[TupleInt3]]],
                       stride_size_per_encoding_block: Optional[Union[TupleInt3, List[TupleInt3]]],
                       reduction_factor: float,
                       expected_num_reduced_features: int,
                       aggregation_type: AggregationType) -> None:
    """
    Test if the image encoder networks can be trained without errors (including GradCam computation and data
    augmentation).
    """
    logging_to_stdout()
    set_random_seed(0)
    dataset_folder = Path(test_output_dirs.make_sub_dir("dataset"))
    scan_size = (6, 64, 60)
    scan_files: List[str] = []
    for s in range(4):
        random_scan = np.random.uniform(0, 1, scan_size)
        scan_file_name = f"scan{s + 1}{NumpyFile.NUMPY.value}"
        np.save(str(dataset_folder / scan_file_name), random_scan)
        scan_files.append(scan_file_name)

    dataset_contents = """subject,channel,path,label,numerical1,numerical2,categorical1,categorical2
S1,week0,scan1.npy,,1,10,Male,Val1
S1,week1,scan2.npy,True,2,20,Female,Val2
S2,week0,scan3.npy,,3,30,Female,Val3
S2,week1,scan4.npy,False,4,40,Female,Val1
S3,week0,scan1.npy,,5,50,Male,Val2
S3,week1,scan3.npy,True,6,60,Male,Val2
"""
    (dataset_folder / "dataset.csv").write_text(dataset_contents)
    numerical_columns = ["numerical1", "numerical2"] if use_non_imaging_features else []
    categorical_columns = ["categorical1", "categorical2"] if use_non_imaging_features else []
    non_image_feature_channels = get_non_image_features_dict(default_channels=["week1", "week0"],
                                                             specific_channels={"categorical2": ["week1"]}) \
        if use_non_imaging_features else {}
    config_for_dataset = ScalarModelBase(
        local_dataset=dataset_folder,
        image_channels=["week0", "week1"],
        image_file_column="path",
        label_channels=["week1"],
        label_value_column="label",
        non_image_feature_channels=non_image_feature_channels,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        should_validate=False
    )
    config_for_dataset.read_dataset_into_dataframe_and_pre_process()

    dataset = ScalarDataset(
        config_for_dataset,
        sample_transform=ScalarItemAugmentation(ImageTransformationPipeline([RandomAffine(10), ColorJitter(0.2)],
                                                     use_different_transformation_per_channel=True)))
    assert len(dataset) == 3

    config = ImageEncoder(
        encode_channels_jointly=encode_channels_jointly,
        should_validate=False,
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        non_image_feature_channels=non_image_feature_channels,
        categorical_feature_encoder=config_for_dataset.categorical_feature_encoder,
        encoder_dimensionality_reduction_factor=reduction_factor,
        aggregation_type=aggregation_type,
        scan_size=(6, 64, 60)
    )

    if kernel_size_per_encoding_block:
        config.kernel_size_per_encoding_block = kernel_size_per_encoding_block
    if stride_size_per_encoding_block:
        config.stride_size_per_encoding_block = stride_size_per_encoding_block

    config.set_output_to(test_output_dirs.root_dir)
    config.max_batch_grad_cam = 1
    model = create_model_with_temperature_scaling(config)
    input_size: List[Tuple] = [(len(config.image_channels), *scan_size)]
    if use_non_imaging_features:
        input_size.append((config.get_total_number_of_non_imaging_features(),))

        # Original number output channels (unreduced) is
        # num initial channel * (num encoder block - 1) = 4 * (3-1) = 8
        if encode_channels_jointly:
            # reduced_num_channels + num_non_img_features
            assert model.final_num_feature_channels == expected_num_reduced_features + \
                   config.get_total_number_of_non_imaging_features()
        else:
            # num_img_channels * reduced_num_channels + num_non_img_features
            assert model.final_num_feature_channels == len(config.image_channels) * expected_num_reduced_features + \
                   config.get_total_number_of_non_imaging_features()

    summarizer = ModelSummary(model)
    summarizer.generate_summary(input_sizes=input_size)
    config.local_dataset = dataset_folder
    config.validate()
    model_train_unittest(config, output_folder=test_output_dirs)
    # No further asserts here because the models are still in experimental state. Most errors would come
    # from having invalid model architectures, which would throw runtime errors during training.


@pytest.mark.gpu
@pytest.mark.parametrize(["encode_channels_jointly", "aggregation_type", "imaging_feature_type"],
                         [(False, AggregationType.Average, ImagingFeatureType.Segmentation),
                          (True, AggregationType.Average, ImagingFeatureType.Segmentation),
                          (False, AggregationType.Average, ImagingFeatureType.ImageAndSegmentation),
                          (True, AggregationType.Average, ImagingFeatureType.ImageAndSegmentation),
                          (True, AggregationType.GatedPooling, ImagingFeatureType.ImageAndSegmentation),
                          (True, AggregationType.MixPooling, ImagingFeatureType.ImageAndSegmentation),
                          (True, AggregationType.ZAdaptive3dAvg, ImagingFeatureType.ImageAndSegmentation)])
def test_image_encoder_with_segmentation(test_output_dirs: OutputFolderForTests,
                                         encode_channels_jointly: bool,
                                         aggregation_type: AggregationType,
                                         imaging_feature_type: ImagingFeatureType) -> None:
    """
    Test if the image encoder networks can be trained on segmentations from HDF5.
    """
    logging_to_stdout()
    set_random_seed(0)
    scan_size = (6, 64, 60)
    dataset_contents = """subject,channel,path,label
    S1,week0,scan1.h5,
    S1,week1,scan2.h5,True
    S2,week0,scan3.h5,
    S2,week1,scan4.h5,False
    S3,week0,scan5.h5,
    S3,week1,scan6.h5,True
    S4,week0,scan7.h5,
    S4,week1,scan8.h5,True
    """
    config = ImageEncoder(encode_channels_jointly=encode_channels_jointly,
                          imaging_feature_type=imaging_feature_type,
                          should_validate=False,
                          aggregation_type=aggregation_type,
                          scan_size=scan_size)
    config.use_mixed_precision = True
    # Necessary because torch otherwise says "avg_pool3d_backward_cuda does not have a deterministic implementation"
    config.pl_deterministic = False
    config.set_output_to(test_output_dirs.root_dir)
    config.num_epochs = 1
    config.local_dataset = Path()
    config.dataset_data_frame = pd.read_csv(StringIO(dataset_contents), sep=",", dtype=str)
    # Patch the load_images function that will be called once we access a dataset item
    image_and_seg = ImageAndSegmentations[np.ndarray](images=np.zeros(scan_size, dtype=np.float32),
                                                      segmentations=np.ones(scan_size, dtype=np.uint8))
    with mock.patch("InnerEye.ML.run_ml.is_offline_run_context", return_value=True):
        with mock.patch('InnerEye.ML.utils.io_util.load_image_in_known_formats', return_value=image_and_seg):
            azure_config = get_default_azure_config()
            azure_config.train = True
            MLRunner(config, azure_config=azure_config).run()
            # No further asserts here because the models are still in experimental state. Most errors would come
            # from having invalid model architectures, which would throw runtime errors during training.
            # Verified manually that the cross entropy on the Val set that appears during training, and the
            # cross entropy when running on the Val set in test mode are the same.


@pytest.mark.parametrize("use_gpu", [True, False] if is_gpu_available() else [False])
@pytest.mark.parametrize("input_on_gpu", [True, False] if is_gpu_available() else [False])
@pytest.mark.gpu
def test_segmentation_to_one_hot(use_gpu: bool, input_on_gpu: bool) -> None:
    # Settings to test on large scale:
    # B = 16
    # C = 2
    # dim = (50, 400, 400)
    B = 2
    C = 3
    dim = (4, 1, 2)
    input_size = (B, C) + dim
    actual_class = 5
    # This is deliberately replicated from get_datatype_for_image_tensors
    dtype = torch.float16 if is_gpu_available() else torch.float32
    device = "cuda" if input_on_gpu else "cpu"
    seg = torch.ones(input_size, dtype=torch.uint8, device=device) * actual_class
    start_time = time.time()
    one_hot = segmentation_to_one_hot(seg, use_gpu, result_dtype=dtype)
    elapsed = time.time() - start_time
    print(f"Computed one-hot in {elapsed:0.2f}sec")
    assert one_hot.shape == (B, C * HDF5_NUM_SEGMENTATION_CLASSES) + dim
    assert one_hot.dtype == dtype
    # The result must be on the same device as the input. In particular, that means we can feed in a CPU
    # tensor, do the computation on the GPU, and still get back a CPU tensor.
    assert seg.device == one_hot.device
    for i in range(C * HDF5_NUM_SEGMENTATION_CLASSES):
        # Dimensions 5, 15, 25 should be all ones
        if i % HDF5_NUM_SEGMENTATION_CLASSES == actual_class:
            expected = torch.ones((B,) + dim, device=one_hot.device)
            assert one_hot[:, i, ...].float().allclose(expected), f"Dimension {i} should have all ones"
        else:
            expected = torch.zeros((B,) + dim, device=one_hot.device)
            assert one_hot[:, i, ...].float().allclose(expected), f"Dimension {i} should have all ones"

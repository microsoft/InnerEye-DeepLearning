#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import random
from typing import Any, Tuple

import pandas as pd

from InnerEye.ML.config import PhotometricNormalizationMethod, SegmentationModelBase, equally_weighted_classes
from InnerEye.ML.deep_learning_config import OptimizerType
from InnerEye.ML.utils.csv_util import CSV_INSTITUTION_HEADER, CSV_SERIES_HEADER, CSV_SUBJECT_HEADER
from InnerEye.ML.utils.split_dataset import DatasetSplits


class ProstatePaperBase(SegmentationModelBase):
    """
    Prostate radiotherapy image segmentation model.
    """

    def __init__(self, **kwargs: Any) -> None:
        fg_classes = ["external", "femur_r", "femur_l", "rectum", "prostate", "bladder", "seminalvesicles"]
        super().__init__(
            should_validate=False,
            architecture="UNet3D",
            feature_channels=[32],
            kernel_size=3,
            crop_size=(64, 224, 224),
            test_crop_size=(128, 512, 512),
            image_channels=["ct"],
            ground_truth_ids=fg_classes,
            largest_connected_component_foreground_classes=["external", "femur_r", "femur_l", "rectum", "prostate",
                                                            "bladder"],

            colours=[(255, 255, 255)] * len(self.fg_classes),
            fill_holes=[False] * len(self.fg_classes),
            ground_truth_ids_display_names=fg_classes,
            num_dataload_workers=8,
            norm_method=PhotometricNormalizationMethod.CtWindow,
            level=50,
            window=600,
            class_weights=equally_weighted_classes(fg_classes, background_weight=0.02),
            train_batch_size=8,
            inference_batch_size=1,
            inference_stride_size=(64, 256, 256),
            start_epoch=0,
            num_epochs=120,
            l_rate=1e-3,
            min_l_rate=1e-5,
            l_rate_polynomial_gamma=0.9,
            optimizer_type=OptimizerType.Adam,
            opt_eps=1e-4,
            adam_betas=(0.9, 0.999),
            momentum=0.9,
            weight_decay=1e-4,
            save_step_epochs=20,
            use_mixed_precision=True,
            use_model_parallel=True,
            monitoring_interval_seconds=0,
        )
        self.add_and_validate(kwargs)

    def get_model_train_test_dataset_splits(self, dataset_df: pd.DataFrame) -> DatasetSplits:
        splits = DatasetSplits.from_institutions(
            df=dataset_df,
            proportion_train=0.8,
            proportion_test=0.1,
            proportion_val=0.1,
            shuffle=True,
            exclude_institutions={
                "ac54f75d-c5fa-4e32-a140-485527f8e3a2",  # Birmingham: 1 image
                "af8d9205-2ae1-422f-8b35-67ee435253e1",  # OSL: 2 images
                "87630c93-07d6-49de-844a-3cc99fe9c323",  # Brussels: 3 images
                "5a6ba8fe-65bc-43ec-b1fc-682c8c37e40c",  # VFN: 4 images
            },
            # These institutions have around 40 images each. The main argument in the paper will be about
            # keeping two of those aside as untouched test sets.
            # Oncoclinicas uses Siemens scanner, IOV uses a GE scanner. Most of the other images are from Toshiba
            # scanners.
            institutions_for_test_only={
                # "d527557d-3b9a-45d0-ad57-692e5a199896",  # AZ Groenige
                "85aaee5f-f5f3-4eae-b6cd-26b0070156d8",  # IOV
                "641eda02-90c3-45ed-b8b1-2651b6a5da6c",  # Oncoclinicas
                # "8522ccd1-ab59-4342-a2ce-7d8ad826ab4f",  # UW
            }
        )

        # IOV subjects not in the test set already
        iov_subjects = {
            "1ec8a7d58cadb231a0412b674731ee72da0e04ab67f2a2f009a768189bbcf691",
            "439bc48993c6e146c4ab573eeba35990ee843b7495dd0924dc6bd0b331e869db",
            "e5d338a12dfcc519787456b09072a07c6191b7140e036c52bc4d039ef3b28afd",
            "af7ad87cc408934cb2a65029661cb426539429a8aada6e1644a67a056c94f691",
            "227e859ee0bd0c4ff860dd77a20f39fe5924348ff4a4fac15dc94cea2cd07c39",
            "512b22856b7dbde60b4a42c348c4bee5b9efb67024fb708addcddfe1f4841288",
            "906f77caba56df060f5d519ae9b6572a90ac22a04560b4d561f3668e6331e3c3",
            "49a01ffe812b0f3e3d93334866662afb5fb33ba6dcd3cc642d4577a449000649",
            "ab3ed87d55da37a2a665b059b5fef54a0553656e8df51592b8c40f16facd60b9",
            "6eb8aeb8f822e15970d3feb64a618a9ad3de936046d84cb83d2569fbb6c70fcb"}

        def _swap_iov(train_val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
            """
            Swap the images that are in the IOV and in the Train/Val, with those from the Test set
            of the same institution (to maintain the institution wise distribution of images)
            """
            random.seed(0)
            # filter iov subjects that are not in the test set (as we do not want to swap them)
            iov_not_in_test = set([x for x in iov_subjects if x not in test_df.seriesId.unique()])

            iov_train_val_subjects = train_val_df[CSV_SERIES_HEADER].isin(iov_not_in_test)
            iov_train_val_subjects_df = train_val_df.loc[iov_train_val_subjects]
            # drop IOV subjects
            train_val_df = train_val_df.loc[~iov_train_val_subjects]
            # select the same number for the same institutions from the test set (ignoring the IOV subjects that
            # are already in the tet set and add it to provided df
            for x in iov_train_val_subjects_df.institutionId.unique():
                test_subs = list(test_df.loc[(test_df[CSV_INSTITUTION_HEADER] == x) & (~test_df[CSV_SERIES_HEADER]
                                                                                       .isin(
                    iov_subjects))].subject.unique())
                num_train_val_df_subs_to_swap = len(
                    iov_train_val_subjects_df.loc[
                        iov_train_val_subjects_df[CSV_INSTITUTION_HEADER] == x].subject.unique())
                subjects_to_swap = random.sample(test_subs, k=num_train_val_df_subs_to_swap)
                # test df to swap
                to_swap = test_df[CSV_SUBJECT_HEADER].isin(subjects_to_swap)
                # swap
                train_val_df = pd.concat([train_val_df, test_df.loc[to_swap]])
                test_df = test_df.loc[~to_swap]

            return train_val_df, test_df

        train_swap, test_swap = _swap_iov(splits.train, splits.test)
        val_swap, test_swap = _swap_iov(splits.val, test_swap)
        test_swap = pd.concat(
            [test_swap, dataset_df.loc[dataset_df[CSV_SERIES_HEADER].isin(iov_subjects)]]).drop_duplicates()

        swapped_splits = DatasetSplits(
            train=train_swap,
            val=val_swap,
            test=test_swap
        )

        iov_intersection = set(swapped_splits.train.seriesId.unique()).intersection(iov_subjects)
        if len(iov_intersection) != 0:
            raise ValueError(f"Train split has IOV subjects {iov_intersection}")
        iov_intersection = set(swapped_splits.val.seriesId.unique()).intersection(iov_subjects)
        if len(iov_intersection) != 0:
            raise ValueError(f"Val split has IOV subjects {iov_intersection}")

        iov_missing = iov_subjects.difference(swapped_splits.test.seriesId.unique())
        if len(iov_missing) != 0:
            raise ValueError(f"All IOV subjects must be in the Test split, found f{iov_missing} that are not")

        def _check_df_distribution(_old_df: pd.DataFrame, _new_df: pd.DataFrame) -> None:
            _old_df_inst = _old_df.drop_duplicates(CSV_SUBJECT_HEADER).groupby([CSV_INSTITUTION_HEADER]).groups
            _new_df_inst = _new_df.drop_duplicates(CSV_SUBJECT_HEADER).groupby([CSV_INSTITUTION_HEADER]).groups
            for k, v in _old_df_inst.items():
                if len(v) != len(_new_df_inst[k]):
                    raise ValueError(f"Expected _new_df to be length={len(v)} found {_new_df_inst[k]}")

        _check_df_distribution(splits.train, swapped_splits.train)
        _check_df_distribution(splits.val, swapped_splits.val)
        _check_df_distribution(splits.test, swapped_splits.test)

        return swapped_splits

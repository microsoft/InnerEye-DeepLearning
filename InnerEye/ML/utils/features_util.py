#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, List, TypeVar

import torch

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.dataset.scalar_dataset import ScalarDataSource, SequenceDataSource
from InnerEye.ML.dataset.sequence_sample import ClassificationItemSequence

FT = TypeVar('FT', ClassificationItemSequence[SequenceDataSource], ScalarDataSource)


@dataclass(frozen=True)
class FeatureStatistics(Generic[FT]):
    """
    Class to store statistics (mean and standard deviation) of a set of features in a given dataset.
    Allows to perform feature standardization for this set of features.
    """
    mean: torch.Tensor  # This tensor will have the same shape as the non-image features in the dataset.
    std: torch.Tensor  # This tensor will have the same shape as the non-image features in the dataset.

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)

    @staticmethod
    def from_data_sources(sources: List[FT]) -> FeatureStatistics:
        """
        For the provided data sources, compute the mean and std across all non-image features across all entries.

        :param sources: list of data sources
        :return: a Feature Statistics object storing mean and standard deviation for each non-imaging feature of
        the dataset.
        """
        if len(sources) == 0:
            raise ValueError("sources must have a length greater than 0")

        data_sources: List[Any]  # for mypy
        if isinstance(sources[0], ClassificationItemSequence):
            data_sources = [item for seq in sources for item in seq.items]
        else:
            data_sources = sources

        numerical_non_image_features = [x.numerical_non_image_features for x in data_sources]
        if len(numerical_non_image_features) == 0:
            raise ValueError("This function must be called with a non-empty set of numerical_non_image_features.")
        unique_shapes = {f.shape for f in numerical_non_image_features}
        if len(unique_shapes) != 1:
            raise ValueError(
                f"All non-image features must have the same size, but got these sizes: {unique_shapes}")

        all_stacked = torch.stack(numerical_non_image_features, dim=0)
        mean = torch.mean(all_stacked, dim=0)
        std = torch.std(all_stacked, dim=0)
        return FeatureStatistics(mean=mean, std=std)

    def standardize(self, sources: List[FT]) -> List[FT]:
        """
        For the provided data sources, apply standardization to the non-image features in each source. This will
        standardize them to mean 0, variance 1 across all sequences.
        All features that have zero standard deviation (constant features) are left untouched.

        :param sources: list of datasources.
        :return list of data sources where all non-imaging features are standardized.
        """

        def apply_source(source: ScalarDataSource) -> ScalarDataSource:
            new_features = (source.numerical_non_image_features - self.mean) / self.std
            zero_or_nan = (self.std == 0.0) + torch.isnan(self.std)
            new_features[zero_or_nan] = source.numerical_non_image_features[zero_or_nan]
            return source.clone_with_overrides(numerical_non_image_features=new_features)

        def apply_sequence(seq: ClassificationItemSequence) -> ClassificationItemSequence:
            # noinspection PyTypeChecker
            return ClassificationItemSequence(id=seq.id, items=list(map(apply_source, seq.items)))

        if len(sources) > 0:
            if isinstance(sources[0], ClassificationItemSequence):
                return list(map(apply_sequence, sources))  # type: ignore
            else:
                return list(map(apply_source, sources))  # type: ignore
        else:
            return sources

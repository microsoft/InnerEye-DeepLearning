#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import torch

from InnerEye.Common.common_util import check_properties_are_not_none
from InnerEye.ML.dataset.scalar_dataset import ScalarDataSource


@dataclass(frozen=True)
class FeatureStatistics:
    """
    Class to store statistics (mean and standard deviation) of a set of features in a given dataset.
    Allows to perform feature standardization for this set of features.
    """
    mean: torch.Tensor  # This tensor will have the same shape as the non-image features in the dataset.
    std: torch.Tensor  # This tensor will have the same shape as the non-image features in the dataset.

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)

    @staticmethod
    def from_data_sources(sources: List[ScalarDataSource]) -> FeatureStatistics:
        """
        For the provided data sources, compute the mean and std across all non-image features across all entries.

        :param sources: list of data sources
        :return: a Feature Statistics object storing mean and standard deviation for each non-imaging feature of
            the dataset.
        """
        if len(sources) == 0:
            raise ValueError("sources must have a length greater than 0")

        data_sources: List[Any] = sources

        numerical_non_image_features = [x.numerical_non_image_features for x in data_sources]
        if len(numerical_non_image_features) == 0:
            raise ValueError("This function must be called with a non-empty set of numerical_non_image_features.")
        unique_shapes = {f.shape for f in numerical_non_image_features}
        if len(unique_shapes) != 1:
            raise ValueError(
                f"All non-image features must have the same size, but got these sizes: {unique_shapes}")

        # If the input features contain infinite values (e.g. from padding)
        # we need to ignore them for the computation of the normalization statistics.
        all_stacked = torch.stack(numerical_non_image_features, dim=0)
        return FeatureStatistics.compute_masked_statistics(input=all_stacked,
                                                           mask=torch.isfinite(all_stacked))

    @staticmethod
    def compute_masked_statistics(input: torch.Tensor, mask: torch.Tensor,
                                  apply_bias_correction: bool = True) -> FeatureStatistics:
        """
        If the input features contains invalid values (e.g. from padding) they should be ignored in the
        computation of the standardization statistics. This function allows to provide a boolean mask (of the same
        shape as the input) to indicate which values should be taken into account for the computation of the
        statistics. All values for which mask == True will be used for computation, the other will be ignored.
        The statistics are computed for each feature i.e. column of the input (shape [batch_size, n_numerical_features])

        :param input: input including all values, of dimension [batch_size, n_numerical_features]
        :param mask: boolean tensor of the same shape as input
        :param apply_bias_correction: if True applies Bessel's correction to the standard deviation estimate
        :return: FeatureStatistics (mean and std) computed on the masked values.
        """
        n_obs_per_feature = mask.sum(dim=0).float()
        masked_values = torch.zeros_like(input)
        masked_values[mask] = input[mask]
        mean = masked_values.sum(dim=0) / n_obs_per_feature
        second_moment = torch.pow(masked_values, 2).sum(dim=0) / n_obs_per_feature
        variance = second_moment - torch.pow(mean, 2)
        if apply_bias_correction:
            # Applies Bessel's bias correction to the std estimate (as in PyTorch's default behavior)
            variance *= torch.div(n_obs_per_feature, (n_obs_per_feature - 1))
        # Need to make sure variance is positive (numerical instability can make it slightly <0)
        std = torch.sqrt(torch.max(variance, torch.zeros_like(variance)))
        return FeatureStatistics(mean=mean, std=std)

    def standardize(self, sources: List[ScalarDataSource]) -> List[ScalarDataSource]:
        """
        For the provided data sources, apply standardization to the non-image features in each source. This will
        standardize them to mean 0, variance 1 across all sequences.
        All features that have zero standard deviation (constant features) are left untouched.

        :param sources: list of datasources.
        :return: list of data sources where all non-imaging features are standardized.
        """

        def apply_source(source: ScalarDataSource) -> ScalarDataSource:
            new_features = (source.numerical_non_image_features - self.mean) / self.std
            zero_or_nan = (self.std == 0.0) + torch.isnan(self.std)
            new_features[zero_or_nan] = source.numerical_non_image_features[zero_or_nan]
            return source.clone_with_overrides(numerical_non_image_features=new_features)

        if len(sources) > 0:
            return list(map(apply_source, sources))  # type: ignore
        else:
            return sources

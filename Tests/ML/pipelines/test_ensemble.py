#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
import torch

from InnerEye.ML.config import EnsembleAggregationType
from InnerEye.ML.pipelines.ensemble import EnsemblePipeline
from InnerEye.ML.pipelines.inference import InferencePipeline
from InnerEye.ML.utils.image_util import posteriors_to_segmentation


def test_aggregate_results() -> None:
    """
    Test to make sure inference results are aggregated as expected
    """
    torch.manual_seed(1)
    num_models = 3
    # set expected posteriors
    model_results = []
    # create results for each model
    for x in range(num_models):
        posteriors = torch.nn.functional.softmax(torch.rand(3, 3, 3, 3), dim=0).numpy()
        model_results.append(InferencePipeline.Result(
            epoch=0,
            patient_id=0,
            posteriors=posteriors,
            segmentation=posteriors_to_segmentation(posteriors),
            voxel_spacing_mm=(1, 1, 1)
        ))

    ensemble_result = EnsemblePipeline.aggregate_results(model_results,
                                                         aggregation_type=EnsembleAggregationType.Average)

    assert ensemble_result.epoch == model_results[0].epoch
    assert ensemble_result.patient_id == model_results[0].patient_id

    expected_posteriors = np.mean([x.posteriors for x in model_results], axis=0)
    assert np.array_equal(ensemble_result.posteriors, expected_posteriors)
    assert np.array_equal(ensemble_result.segmentation, posteriors_to_segmentation(expected_posteriors))

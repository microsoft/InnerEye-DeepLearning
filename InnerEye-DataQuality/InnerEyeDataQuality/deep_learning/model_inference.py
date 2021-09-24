#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import Any, List, Tuple

import numpy as np
from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.collect_embeddings import get_all_embeddings, register_embeddings_collector
from InnerEyeDataQuality.deep_learning.utils import get_run_config
from InnerEyeDataQuality.deep_learning.trainers.vanilla_trainer import VanillaTrainer
from InnerEyeDataQuality.deep_learning.trainers.model_trainer_base import ModelTrainer
from InnerEyeDataQuality.deep_learning.trainers.co_teaching_trainer import CoTeachingTrainer
from InnerEyeDataQuality.deep_learning.dataloader import get_val_dataloader
from InnerEyeDataQuality.utils.custom_types import SelectorTypes as ST

NUM_MULTIPLE_RUNS = 10

def inference_model(dataloader: Any, model_trainer: ModelTrainer, use_mc_sampling: bool = False) -> Tuple[List, List]:
    """
    Performs an inference pass on a single model
    :param config:
    :return:
    """
    # Inference on given dataloader
    all_model_cnn_embeddings = register_embeddings_collector(model_trainer.models, use_only_in_train=False)
    trackers = model_trainer.run_inference(dataloader, use_mc_sampling)
    embs = get_all_embeddings(all_model_cnn_embeddings)
    probs = [metric_tracker.sample_metrics.probabilities for metric_tracker in trackers]
    return embs, probs


def inference_ensemble(dataset: Any, config: ConfigNode) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ModelTrainer]:
    """
    Returns: 
        embeddings: 2D numpy array - containing sample embeddings obtained from a CNN.
                    [num_samples, embedding_size]
        posteriors: 2D numpy array - containing class posteriors obtained from a CNN.
                    [num_samples, num_classes]
        trainer:    Model trainer object built using config file
    """

    # Reload the model from config
    model_trainer_class = CoTeachingTrainer if config.train.use_co_teaching else VanillaTrainer
    config = get_run_config(config, config.train.seed)
    model_trainer = model_trainer_class(config=config)
    model_trainer.load_checkpoints(restore_scheduler=False)

    # Prepare output data structures
    all_embeddings = []
    all_posteriors = []

    # Run inference on the given dataset
    use_mc_sampling = config.model.use_dropout and ST(config.selector.type[0]) == ST.BaldSelector
    multi_inference = config.train.use_self_supervision or config.model.use_dropout
    num_runs = NUM_MULTIPLE_RUNS if multi_inference else 1
    for run_ind in range(num_runs):
        dataloader = get_val_dataloader(dataset, config, seed=config.train.seed + run_ind) 
        embeddings, posteriors = inference_model(dataloader, model_trainer, use_mc_sampling)
        all_embeddings.append(embeddings)
        all_posteriors.append(posteriors)

    # Aggregate results and return
    embeddings = np.mean(all_embeddings, axis=(0, 1))
    all_posteriors = np.stack(all_posteriors, axis=0)
    avg_posteriors = np.mean(all_posteriors, axis=(0, 1))

    return embeddings, avg_posteriors, all_posteriors, model_trainer

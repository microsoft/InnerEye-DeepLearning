#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.model_inference import inference_ensemble
from InnerEyeDataQuality.deep_learning.utils import load_selector_config
from InnerEyeDataQuality.selection.selectors.base import SampleSelector
from InnerEyeDataQuality.selection.selectors.bald import BaldSelector
from InnerEyeDataQuality.selection.selectors.graph import GraphBasedSelector, GraphParameters
from InnerEyeDataQuality.selection.selectors.label_based import LabelBasedDecisionRule, PosteriorBasedSelector

from InnerEyeDataQuality.selection.simulation_statistics import get_ambiguous_sample_ids
from InnerEyeDataQuality.utils.custom_types import SelectorTypes as ST
from InnerEyeDataQuality.utils.plot import plot_model_embeddings


def evaluate_ambiguous_case_detection(bald_score: np.ndarray, labels_complete: np.ndarray) -> None:
    uncertain_cases = np.zeros_like(bald_score)
    true_ambiguous_cases = get_ambiguous_sample_ids(labels_complete)
    uncertain_cases[true_ambiguous_cases] = 1
    auc_ = roc_auc_score(y_true=uncertain_cases, y_score=bald_score)
    logging.info(f'BALD ambiguous detection AUC: {float(auc_):.2f}')

def pretty_selector_name(_type: str, model_name: str) -> str:
    type_dict = {'BaldSelector': None,
                 'PosteriorBasedSelector': None,
                 'PosteriorBasedSelectorJoint': 'With entropy',
                 'GraphBasedSelector': 'Graph'}
    _type = type_dict[_type]  # type: ignore
    return f'{model_name} ({_type})' if _type else f'{model_name}'


def get_selector(_type: str, cfg: ConfigNode, **pars: Any) -> SampleSelector:
    name = pars["name"]
    num_samples = pars["dataset"].num_samples
    num_classes = pars["dataset"].num_classes
    sample_indices = pars["dataset"].indices
    embeddings = pars["embeddings"]
    avg_posteriors = pars["avg_posteriors"]
    all_posteriors = pars["all_posteriors"]
    output_directory = pars["output_directory"]
    trainer = pars["trainer"]
    use_active_relabelling = pars["use_active_relabelling"]

    if ST(_type) is ST.GraphBasedSelector:
        distance_metric = "cosine" if (
                cfg.model.resnet.apply_l2_norm or cfg.train.use_self_supervision) else "euclidean"
        graph_params = GraphParameters(n_neighbors=num_samples // 200,
                                       diffusion_alpha=0.90,
                                       cg_solver_max_iter=10,
                                       diffusion_batch_size=num_samples // 200,
                                       distance_kernel=distance_metric)
        return GraphBasedSelector(num_samples, num_classes, embeddings,
                                  sample_indices=sample_indices, name=name,
                                  graph_params=graph_params)

    elif ST(_type) is ST.BaldSelector:
        return BaldSelector(posteriors=all_posteriors,
                            num_samples=num_samples,
                            num_classes=num_classes,
                            name=name,
                            trainer=trainer,
                            use_active_relabelling=use_active_relabelling)

    elif ST(_type) is ST.PosteriorBasedSelector:
        return PosteriorBasedSelector(avg_posteriors, num_samples, num_classes=num_classes, name=name,
                                      decision_rule=LabelBasedDecisionRule.CROSS_ENTROPY,
                                      output_directory=output_directory, trainer=trainer,
                                      use_active_relabelling=use_active_relabelling)

    elif ST(_type) is ST.PosteriorBasedSelectorJoint:
        return PosteriorBasedSelector(avg_posteriors, num_samples, num_classes=num_classes, name=name,
                                      decision_rule=LabelBasedDecisionRule.JOINT,
                                      output_directory=output_directory, trainer=trainer,
                                      use_active_relabelling=use_active_relabelling)

    else:
        raise ValueError("Unknown selector type is specified")


def get_user_specified_selectors(list_configs: List[str],
                                 dataset: Any,
                                 output_path: Path,
                                 plot_embeddings: bool = False) -> Dict[str, SampleSelector]:
    """
    Load the user specific configs, get the embeddings and return the selectors.
    :param list_configs:
    :return: dictionary of selector
    """
    logging.info("Loading the selector configs:\n {0}".format('\n'.join(list_configs)))
    user_specified_selectors = dict()
    for cfg in [load_selector_config(cfg) for cfg in list_configs]:
        # Collect model probability predictions for the given set of images in the training set.
        embeddings, avg_posteriors, all_posteriors, trainer = inference_ensemble(dataset, cfg)
        assert avg_posteriors.shape[0] == dataset.num_samples

        if plot_embeddings:
            sample_label_counts = dataset.label_counts
            plot_model_embeddings(embeddings=embeddings, label_distribution=sample_label_counts,
                                  label_names=dataset.get_label_names(), save_path=output_path)

        for _type in cfg.selector.type:
            selector_params = {"dataset": dataset,
                               "trainer": trainer if cfg.selector.use_active_relabelling else None,
                               "embeddings": embeddings,
                               "avg_posteriors": avg_posteriors,
                               "all_posteriors": all_posteriors,
                               "output_directory": cfg.selector.output_directory,
                               "use_active_relabelling": cfg.selector.use_active_relabelling,
                               "name": pretty_selector_name(_type, cfg.selector.model_name)}
            selector_name = pretty_selector_name(_type, cfg.selector.model_name)
            user_specified_selectors[selector_name] = get_selector(_type, cfg, **selector_params)

    return user_specified_selectors


def update_trainer_for_simulation(selector: Any, seed: int) -> None:
    if selector.trainer is None:
        return

    # check if device_id is within the range
    num_gpus = torch.cuda.device_count()
    device_id = seed % num_gpus

    # set the device attribute in config object
    selector.trainer.config.defrost()
    selector.trainer.config.device = device_id
    selector.trainer.config.train.seed = seed
    selector.trainer.config.train.dataloader.num_workers = 0
    selector.trainer.config.validation.dataloader.num_workers = 0
    selector.trainer.config.freeze()
    selector.trainer.device = torch.device(device_id)

    # migrate all parameters to the given device
    selector.trainer.models = [model.to(device_id) for model in selector.trainer.models]

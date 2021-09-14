#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import argparse
import logging
import os

from InnerEyeDataQuality.configs.config_node import ConfigNode
from InnerEyeDataQuality.deep_learning.utils import create_logger, get_run_config, load_model_config
from InnerEyeDataQuality.deep_learning.trainers.co_teaching_trainer import CoTeachingTrainer
from InnerEyeDataQuality.deep_learning.trainers.elr_trainer import ELRTrainer
from InnerEyeDataQuality.deep_learning.trainers.vanilla_trainer import VanillaTrainer
from InnerEyeDataQuality.utils.generic import set_seed


def train(config: ConfigNode) -> None:
    create_logger(config.train.output_dir)
    logging.info('Starting training...')
    if config.train.use_co_teaching and config.train.use_elr:
        raise ValueError("You asked for co-teaching and ELR at the same time. Please double check your configuration.")
    if config.train.use_co_teaching:
        model_trainer_class = CoTeachingTrainer
    elif config.train.use_elr:
        model_trainer_class = ELRTrainer  # type: ignore
    else:
        model_trainer_class = VanillaTrainer  # type: ignore
    model_trainer_class(config).run_training()


def train_ensemble(config: ConfigNode, num_runs: int) -> None:
    for i, _ in enumerate(range(num_runs)):
        config_run = get_run_config(config, config.train.seed + i)
        set_seed(config_run.train.seed)
        os.makedirs(config_run.train.output_dir, exist_ok=True)
        train(config_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TBD')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file characterising trained CNN model/s')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of runs (ensemble)')
    args, unknown_args = parser.parse_known_args()

    # Load config
    config = load_model_config(args.config)

    # Launch training
    train_ensemble(config, args.num_runs)

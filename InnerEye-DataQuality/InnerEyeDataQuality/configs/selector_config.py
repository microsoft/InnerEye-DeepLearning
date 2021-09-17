#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from InnerEyeDataQuality.configs.config_node import ConfigNode

config = ConfigNode()

# data selector
config.selector = ConfigNode()
config.selector.type = None
config.selector.model_name = None
config.selector.model_config_path = None
config.selector.use_active_relabelling = False

# Other selector parameters (unused)
config.selector.training_dynamics_data_path = None
config.selector.burnout_period = 0
config.selector.number_samples_to_relabel = 10

# output files
config.selector.output_directory = None

# tensorboard
config.tensorboard = ConfigNode()
config.tensorboard.save_events = False

def get_default_selector_config() -> ConfigNode:
    return config.clone()

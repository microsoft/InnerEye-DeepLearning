#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from InnerEye.Common.common_util import MAX_PATH_LENGTH, check_properties_are_not_none, is_long_path


@dataclass_json
@dataclass
class ModelInferenceConfig:
    """Class for configuring a model for inference"""
    model_name: str
    checkpoint_paths: List[str]
    model_configs_namespace: str = ''

    def __post_init__(self) -> None:
        check_properties_are_not_none(self)
        # check to make sure paths are no long paths are provided
        long_paths = list(filter(is_long_path, self.checkpoint_paths))
        if long_paths:
            raise ValueError(f"Following paths: {long_paths} are greater than {MAX_PATH_LENGTH}")

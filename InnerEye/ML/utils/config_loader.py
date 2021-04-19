#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import importlib
import inspect
import logging
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional

import param
from importlib._bootstrap import ModuleSpec

from InnerEye.Common.common_util import path_to_namespace
from InnerEye.Common.generic_parsing import GenericConfig
from InnerEye.ML.deep_learning_config import DeepLearningConfig


class ModelConfigLoader(GenericConfig):
    """
    Helper class to manage model config loading
    """
    model_configs_namespace: Optional[str] = param.String(default=None,
                                                          doc="Non-default namespace to search for model configs")

    def __init__(self, **params: Any):
        super().__init__(**params)
        default_module = self.get_default_search_module()
        self.module_search_specs: List[ModuleSpec] = [importlib.util.find_spec(default_module)]
        if self.model_configs_namespace and self.model_configs_namespace != default_module:
            # The later member of this list will take priority if a model name occurs in both, because
            # dict.update is used to combine the dictionaries of models.
            custom_spec = importlib.util.find_spec(self.model_configs_namespace)
            if custom_spec is None:
                raise ValueError(f"Search namespace {self.model_configs_namespace} was not found.")
            self.module_search_specs.append(custom_spec)

    @staticmethod
    def get_default_search_module() -> str:
        from InnerEye.ML import configs
        return configs.__name__

    def create_model_config_from_name(self, model_name: str) -> DeepLearningConfig:
        """
        Returns a model configuration for a model of the given name. This can be either a segmentation or
        classification configuration for an InnerEye built-in model, or a LightningContainer.
        To avoid having to import torch here, there are no references to LightningContainer.
        Searching for a class member called <model_name> in the search modules provided recursively.

        :param model_name: Name of the model for which to get the configs for.
        """
        if not model_name:
            raise ValueError("Unable to load a model configuration because the model name is missing.")

        configs: Dict[str, DeepLearningConfig] = {}

        def _get_model_config(module_spec: ModuleSpec) -> Optional[DeepLearningConfig]:
            """
            Given a module specification check to see if it has a class property with
            the <model_name> provided, and instantiate that config class with the
            provided <config_overrides>. Otherwise, return None.
            :param module_spec:
            :return: Instantiated model config if it was found.
            """
            # noinspection PyBroadException
            try:
                logging.debug(f"Importing {module_spec.name}")
                target_module = importlib.import_module(module_spec.name)
                # The "if" clause checks that obj is a class, of the desired name, that is
                # defined in this module rather than being imported into it (and hence potentially
                # being found twice).
                _class = next(obj for name, obj in inspect.getmembers(target_module)
                              if inspect.isclass(obj)
                              and name == model_name
                              and inspect.getmodule(obj) == target_module)
                logging.info(f"Found class {_class} in file {module_spec.origin}")
            # ignore the exception which will occur if the provided module cannot be loaded
            # or the loaded module does not have the required class as a member
            except Exception as e:
                exception_text = str(e)
                if exception_text != "":
                    logging.warning(f"(from attempt to import module {module_spec.name}): {exception_text}")
                return None
            model_config: DeepLearningConfig = _class()
            return model_config

        def _search_recursively_and_store(module_search_spec: ModuleSpec) -> None:
            """
            Given a root namespace eg: A.B.C searches recursively in all child namespaces
            for class property with the <model_name> provided. If found, this is
            instantiated with the provided overrides, and added to the configs dictionary.
            """
            root_namespace = module_search_spec.name
            namespaces_to_search: List[str] = []
            if module_search_spec.submodule_search_locations:
                # There is little documentation about ModuleSpec, and in particular how the submodule search locations
                # are structured. From the examples I saw, the _path field usually has two entries that only differ by
                # case and/or directory separator. For ambiguous paths, there may be more search locations.
                logging.debug(f"Searching through {len(module_search_spec.submodule_search_locations)} folders that "
                              f"match namespace {module_search_spec.name}: "
                              f"{module_search_spec.submodule_search_locations}")
                for root in module_search_spec.submodule_search_locations:
                    for n in Path(root).rglob("*"):
                        if n.is_file() and "__pycache__" not in str(n):
                            sub_namespace = path_to_namespace(n, root=root)
                            namespaces_to_search.append(root_namespace + "." + sub_namespace)
            elif module_search_spec.origin:
                # The module search spec already points to a python file: Search only that.
                namespaces_to_search.append(module_search_spec.name)
            else:
                raise ValueError(f"Unable to process module spec: {module_search_spec}")

            for n in namespaces_to_search:  # type: ignore
                _module_spec = None
                # noinspection PyBroadException
                try:
                    _module_spec = find_spec(n)  # type: ignore
                except Exception:
                    pass

                if _module_spec:
                    config = _get_model_config(_module_spec)
                    if config:
                        configs[n] = config  # type: ignore

        for search_spec in self.module_search_specs:
            _search_recursively_and_store(search_spec)

        if len(configs) == 0:
            raise ValueError(
                f"Model name {model_name} was not found in search namespaces: "
                f"{[s.name for s in self.module_search_specs]}.")
        elif len(configs) > 1:
            raise ValueError(
                f"Multiple instances of model name {model_name} were found in namespaces: {configs.keys()}.")
        else:
            return list(configs.values())[0]

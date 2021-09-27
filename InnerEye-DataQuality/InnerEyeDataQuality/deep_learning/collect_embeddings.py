#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from typing import List, Optional, Union

import numpy as np
import torch
from InnerEyeDataQuality.deep_learning.self_supervised.ssl_classifier_module import PretrainedClassifier, SSLClassifier


class CollectCNNEmbeddings:
    """
    This class takes care of registering a forward hook to get the embeddings for a given model.
    """

    def __init__(self, use_only_in_train: bool, store_input_tensor: bool) -> None:
        """
        :param use_only_in_train: If set to True, hooks are registered only for model forward passes in training mode.
        :param store_input_tensor:
        """
        self.inputs: list = list()
        self.layer: Optional[torch.nn.Module] = None
        self.use_only_in_train = use_only_in_train
        self.store_input_tensor = store_input_tensor

    def __call__(self,
                 module: torch.nn.Module,
                 module_in: List[torch.Tensor],
                 module_out: torch.Tensor) -> None:

        # If model is in validation state and only training time collection is allowed, then exit.
        if self.use_only_in_train and not module.training:
            return
        if module.use_hook:
            _tensor = module_in[0] if self.store_input_tensor else module_out
            self.inputs.append(_tensor.detach().cpu())

    def return_embeddings(self, return_numpy: int = True) -> Union[np.ndarray, torch.Tensor, None]:
        if len(self.inputs) == 0:
            return None
        embeddings = torch.cat(self.inputs, dim=0)
        return embeddings.cpu().numpy() if return_numpy else embeddings

    def reset(self) -> None:
        self.inputs = list()


def register_embeddings_collector(models: List[torch.nn.Module],
                                  use_only_in_train: bool = False) -> List[CollectCNNEmbeddings]:
    """
    Takes a list of models and register a foward hook for each model
    :param models: Torch module
    :param use_only_in_train: If set to True, hooks are registered only for model forward passes in training mode.
    """
    assert(isinstance(use_only_in_train, bool))
    all_model_cnn_embeddings = []
    for model in models:
        store_input_tensor = False if isinstance(model, SSLClassifier) else True
        cnn_embeddings = CollectCNNEmbeddings(use_only_in_train, store_input_tensor)
        if hasattr(model, "projection"):
            cnn_embeddings.layer = model.projection  # type: ignore
        elif hasattr(model, "resnet"):
            cnn_embeddings.layer = model.resnet.fc  # type: ignore
        elif isinstance(model, PretrainedClassifier):
            cnn_embeddings.layer = model.classifier_head
        else:
            cnn_embeddings.layer = model.fc  # type: ignore
        cnn_embeddings.layer.use_hook = True  # type: ignore
        cnn_embeddings.layer.register_forward_hook(cnn_embeddings)  # type: ignore
        all_model_cnn_embeddings.append(cnn_embeddings)
    return all_model_cnn_embeddings


def get_all_embeddings(embeddings_collectors: List[CollectCNNEmbeddings]) -> List[np.ndarray]:
    """
    Returns all embeddings from a list of embeddings collectors and resets the list
    """
    output = list()
    for cnn_embeddings in embeddings_collectors:
        output.append(cnn_embeddings.return_embeddings(return_numpy=True))
        cnn_embeddings.reset()
    return output

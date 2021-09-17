#  ------------------------------------------------------------------------------------------
#  Copyright (c) 2017 hysts.
#  Licensed under the MIT License (MIT). See LICENSE in the PytrochImageClassification folder for license information.
#  ------------------------------------------------------------------------------------------

import torch

def get_param_list(config, model):
    if config.train.no_weight_decay_on_bn:
        param_list = []
        for name, params in model.named_parameters():
            if 'conv.weight' in name:
                param_list.append({
                    'params': params,
                    'weight_decay': config.train.weight_decay,
                })
            else:
                param_list.append({
                    'params': params,
                    'weight_decay': 0,
                })
    else:
        param_list = [{
            'params': list(model.parameters()),
            'weight_decay': config.train.weight_decay,
        }]
    return param_list


def create_optimizer(config, model):
    params = get_param_list(config, model)

    if config.train.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params,
                                    lr=config.train.base_lr,
                                    momentum=config.train.momentum,
                                    nesterov=config.train.nesterov)
    elif config.train.optimizer == 'adam':
        optimizer = torch.optim.Adam(params,
                                     lr=config.train.base_lr,
                                     betas=config.optim.adam.betas)
    else:
        raise ValueError()
    return optimizer

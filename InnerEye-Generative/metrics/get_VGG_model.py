#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import torch
import torchvision

class SmallOutputVGG(torch.nn.Module):
    def __init__(self, model, dim_in=1,  dim_out=2048):
        super().__init__()
        model.features[0] = torch.nn.Conv2d(dim_in, 64, kernel_size=3, stride=1, padding=1)
        model.classifier = torch.nn.Linear(512 * 7 * 7, dim_out)
        self.model = model

    def forward(self, x):
        return self.model.forward(x)


def load_model(dim_in=1):
    model = torchvision.models.vgg11(pretrained=False, progress=True)
    model = SmallOutputVGG(model, dim_in)
    model.eval()
    return model

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import numpy as np

from torch import nn
from typing import Tuple, Dict, Optional, List

from InnerEye.ML.models.layers.weight_standardization import WeightStandardizedConv2d
from InnerEye.ML.dataset.scalar_sample import ScalarItem
from InnerEye.ML.utils.device_aware_module import DeviceAwareModule


class ResNetV2Block(nn.Module):
    """
    ResNetV2 (https://arxiv.org/pdf/1603.05027.pdf) uses pre activation in the ResNet blocks.
    Big Transfer replaces BatchNorm with GroupNorm
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_channels: int,
                 num_groups: int,
                 downsample_stride: int = 1):
        super().__init__()

        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = WeightStandardizedConv2d(in_channels=in_channels,
                                              out_channels=bottleneck_channels,
                                              kernel_size=(1, 1),
                                              stride=(1, 1),
                                              bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=bottleneck_channels)
        self.conv2 = WeightStandardizedConv2d(in_channels=bottleneck_channels,
                                              out_channels=bottleneck_channels,
                                              kernel_size=(3, 3),
                                              stride=(downsample_stride, downsample_stride),
                                              padding=(1, 1),
                                              bias=False)
        self.gn3 = nn.GroupNorm(num_groups=num_groups, num_channels=bottleneck_channels)
        self.conv3 = WeightStandardizedConv2d(in_channels=bottleneck_channels,
                                              out_channels=out_channels,
                                              kernel_size=(1, 1),
                                              stride=(1, 1),
                                              bias=False)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.downsample: Optional[WeightStandardizedConv2d] = \
                                        WeightStandardizedConv2d(in_channels=in_channels,
                                                                 out_channels=out_channels,
                                                                 kernel_size=(1, 1),
                                                                 stride=(downsample_stride, downsample_stride),
                                                                 bias=False)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        input_normed = self.relu(self.gn1(x))

        out = self.conv1(input_normed)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        # add residual
        residual = self.downsample(input_normed) if self.downsample else input_normed

        return out + residual


class ResNetV2Layer(nn.Module):
    """
    Single layer of ResNetV2
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 bottleneck_channels: int,
                 num_groups: int,
                 downsample_stride: int,
                 num_blocks: int):
        super().__init__()
        _layers = [ResNetV2Block(in_channels=in_channels if i == 0 else out_channels,
                                 out_channels=out_channels,
                                 bottleneck_channels=bottleneck_channels,
                                 num_groups=num_groups,
                                 downsample_stride=downsample_stride if i == 0 else 1)
                   for i in range(num_blocks)]

        self.layer = nn.Sequential(*_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.layer(x)


class BiTResNetV2(nn.Module):
    """
    Implements the Big Transfer (BiT) model

    https://arxiv.org/pdf/1912.11370.pdf
    https://github.com/google-research/big_transfer
    """
    def __init__(self, num_groups: int = 32,
                 num_classes: int = 21843,
                 num_blocks_in_layer: Tuple[int, int, int, int] = (3, 4, 23, 3),
                 width_factor: int = 1):
        super().__init__()
        self.initial = nn.Sequential(
                            WeightStandardizedConv2d(in_channels=3,
                                                     out_channels=64 * width_factor,
                                                     kernel_size=(7, 7),
                                                     stride=(2, 2),
                                                     padding=(3, 3),
                                                     bias=False),
                            nn.ConstantPad2d(padding=1, value=0),
                            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1))

        self.layer1 = ResNetV2Layer(in_channels=64 * width_factor,
                                    out_channels=256 * width_factor,
                                    bottleneck_channels=64 * width_factor,
                                    num_groups=num_groups,
                                    downsample_stride=1,
                                    num_blocks=num_blocks_in_layer[0])
        self.layer2 = ResNetV2Layer(in_channels=256 * width_factor,
                                    out_channels=512 * width_factor,
                                    bottleneck_channels=128 * width_factor,
                                    num_groups=num_groups,
                                    downsample_stride=2,
                                    num_blocks=num_blocks_in_layer[1])
        self.layer3 = ResNetV2Layer(in_channels=512 * width_factor,
                                    out_channels=1024 * width_factor,
                                    bottleneck_channels=256 * width_factor,
                                    num_groups=num_groups,
                                    downsample_stride=2,
                                    num_blocks=num_blocks_in_layer[2])
        self.layer4 = ResNetV2Layer(in_channels=1024 * width_factor,
                                    out_channels=2048 * width_factor,
                                    bottleneck_channels=512 * width_factor,
                                    num_groups=num_groups,
                                    downsample_stride=2,
                                    num_blocks=num_blocks_in_layer[3])

        self.linear = nn.Sequential(nn.GroupNorm(num_groups=num_groups, num_channels=2048 * width_factor),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(output_size=1),
                                    nn.Conv2d(in_channels=2048 * width_factor,
                                              out_channels=num_classes,
                                              kernel_size=(1, 1),
                                              bias=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.linear(x)
        return x

    @classmethod
    def convert_weights_to_state_dict(cls, weight_dict: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """
        Converts the pretrained weights supplied by the authors to a state dict which is compatible with
        this implementation.
        """
        def map_name(weight_name: str) -> str:
            constructed = ""
            parts = name.split("/")[1:]
            if parts[0].startswith("block"):
                constructed += parts[0].replace("block", "layer") + "."

                # Converts, for example, unit02 to layer.1.
                constructed += f"layer.{str(int(parts[1][4:].lstrip('0'))-1)}."

                if parts[3] == "group_norm":
                    constructed += "gn"
                elif parts[3] == "standardized_conv2d":
                    constructed += "conv"
                elif parts[3] == "proj":
                    constructed += "downsample"
                else:
                    raise ValueError(f"Unknown layer {weight_name}")

                if not constructed.endswith("downsample"):
                    constructed += str(ord(parts[2][0]) - 96)

            elif parts[0] == "root_block" and parts[1] == "standardized_conv2d":
                constructed = "initial.0"
            elif parts[0] == "group_norm":
                constructed = "linear.0"
            elif parts[0] == "head" and parts[1] == "conv2d":
                constructed = "linear.3"
            else:
                raise ValueError(f"Unknown layer {weight_name}")

            if parts[-1] == "gamma":
                constructed += ".weight"
            elif parts[-1] == "beta":
                constructed += ".bias"
            elif parts[-1] == "kernel":
                constructed += ".weight"
            elif parts[-1] == "bias":
                constructed += ".bias"
            else:
                raise ValueError(f"Unknown layer {weight_name}")

            return constructed

        state_dict = dict()
        for name, weight in weight_dict.items():
            tensor = torch.from_numpy(weight)
            if tensor.ndim == 4:  # type: ignore
                tensor = tensor.permute(3, 2, 0, 1)
            state_dict[map_name(name)] = tensor

        return state_dict


class BiTModel(DeviceAwareModule[ScalarItem, torch.Tensor]):
    def __init__(self):
        super().__init__()
        self.model = BiTResNetV2()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        return self.model(x)

    def get_input_tensors(self, item: ScalarItem) -> List[torch.Tensor]:
        return [item.images]

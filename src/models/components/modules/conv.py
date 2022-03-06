
from typing import Union
import torch
import torch.nn as nn

from src.utils.torch_utils import Activation


class Conv(nn.Module):

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        groups: int = 1,
        activation: Union[str, None] = "ReLU",
    ) -> None:

        super().__init__()

        self.conv = nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride,
            padding=padding,
            groups=groups,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = Activation(activation)()
    
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Forward"""
        return self.act(self.bn(self.conv(x)))

    """without bacthnorm"""
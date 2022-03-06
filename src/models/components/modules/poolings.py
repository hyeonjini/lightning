
import torch
import torch.nn as nn

class MaxPool(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        padding: int
    ):
        super

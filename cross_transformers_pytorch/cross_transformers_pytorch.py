import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

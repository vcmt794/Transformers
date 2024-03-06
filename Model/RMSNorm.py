import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # Use this nn.Parameter to assign
        # (tell model that this weight is learnable)
        self.eps = eps

    def forward(self, inputs):
        hidden = inputs.to(torch.float32)
        variance = torch.pow(hidden, 2).mean(dim=-1)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden).to(inputs.dtype)

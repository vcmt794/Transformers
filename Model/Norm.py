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
        variance = torch.pow(hidden, 2).mean(dim=-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden).to(inputs.dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps: float):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, inputs):
        hidden = inputs.to(torch.float32)
        variance = torch.pow(hidden, 2).mean(dim=-1, keepdim=True)
        hidden = (hidden - torch.mean(hidden))*torch.rsqrt(variance+self.eps)
        return (self.gamma * hidden - self.bias).to(inputs.dtype)

# but I recommend using torch.nn.LayerNorm if u insist on using LayerNorm

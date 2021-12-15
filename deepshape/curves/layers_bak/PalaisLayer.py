import torch
import numpy as np
from torch.nn import Parameter

from .DeepShapeLayer import DeepShapeLayer
from ...common.derivatives import central_differences


class PalaisLayer(DeepShapeLayer):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1, dtype=torch.float)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.project(p=2)

    def forward(self, x):
        return x + (torch.sin(np.pi * self.nvec * x) / (np.pi * self.nvec)) @ self.weights

    def derivative(self, x, h=None):
        return 1. + torch.cos(np.pi * self.nvec * x) @ self.weights

    def project(self, p=1):
        with torch.no_grad():
            if p == 1:
                norm = self.weights.norm(p)
            elif p == float('inf'):
                norm = torch.abs(self.weights).max() * self.N
            elif p == 2:
                norm = self.weights.norm(p) * np.sqrt(self.N)
            else:
                q = p / (p - 1)
                norm = self.weights.norm(p) * torch.ones(self.N).norm(q)
            if norm > 1.0 - 1e-4:
                self.weights *= (1 - 1e-4) / norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self

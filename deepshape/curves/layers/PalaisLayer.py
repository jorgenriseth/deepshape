import torch
import numpy as np
from torch.nn import Parameter

from .DeepShapeLayer import DeepShapeLayer
   
class PalaisLayer(DeepShapeLayer):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1, dtype=torch.float)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.project()
        
    def forward(self, x):
        z = torch.sin(np.pi * self.nvec * x) / (self.nvec * np.pi)
        return x + z @ self.weights
    
    def derivative(self, x, h=None):
        y = torch.cos(np.pi * self.nvec * x)
        return 1. + y @ self.weights
    
    def project(self):
        with torch.no_grad():
            norm = self.weights.norm()
            if norm > 0.99:
                self.weights /= norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self

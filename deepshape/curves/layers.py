import torch
import torch.nn as nn

import numpy as np
from numpy import pi


# Define a Fourier Sine Layer    
class FourierLayer(nn.Module):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.find_ymin()
        self.project()
        
    def forward(self, x):
        z = np.pi * self.nvec * x
        y = np.pi * self.nvec * torch.cos(z)
        z = torch.sin(z)
        self.ymin = torch.min(y).item()
        return x + z @ self.weights, 1. + y @ self.weights

    def find_ymin(self, npoints=1024):
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)
        self.K = npoints
        
        _, y = self.forward(x)
        self.ymin = torch.min(y).item()
        return self.ymin
    
    def project(self, npoints=1024, epsilon=None, stabilizer=1e-4):
        self.find_ymin(npoints)

        if epsilon is None:
            # epsilon = torch.norm(self.weights, 1) * self.N**2 / (8 * self.K)
            epsilon = stabilizer + self.weights.norm(1) * ( 0.5 * self.N * np.pi / npoints)**3
            
        with torch.no_grad():
            if self.ymin < epsilon:
                self.weights *= 1 / (1 + epsilon - self.ymin)


# Define a Palais Layer    
class PalaisLayer(nn.Module):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.find_ymin()
        self.project()
        
    def forward(self, x):
        z = np.pi * self.nvec * x
        y = np.pi * torch.cos(z)
        z = torch.sin(z) / self.nvec
        self.ymin = torch.min(y).item()
        return x + z @ self.weights, 1. + y @ self.weights

    def find_ymin(self, npoints=1024):
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)
        self.K = npoints
        
        _, y = self.forward(x)
        self.ymin = torch.min(y).item()
        return self.ymin
    
    def project(self, npoints=1024, epsilon=None):
        self.find_ymin(npoints)

        if epsilon is None:
            epsilon = torch.norm(self.weights, 1) * self.N**2 / (8 * self.K)
            
        with torch.no_grad():
            if self.ymin < epsilon:
                self.weights *= 1 / (1 + epsilon - self.ymin)
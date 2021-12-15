""" Various layers kept for legacy-purposes. Will probably be removed at some point."""
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


class BumpLayer(nn.Module):
    def forward(self, x):
        return nn.ReLU(x) * nn.ReLU(1-x)


class BumpReluLayer(nn.Module):
    def __init__(self, N, init_scale=1.):
        super().__init__()
        self.N = N
        self.weights1 = torch.nn.Parameter(1. * torch.randn(1, N))
        self.weights2 = torch.nn.Parameter(init_scale * torch.randn(N, 1))
        self.bias = torch.nn.Parameter(
            -self.weights1 * torch.rand(self.N)
        )
        # print(self.bias / (- self.weights1))
        # self.project()


    def bump(self, x):
        return torch.relu(x) * torch.relu(1 - x)

    def heaviside(self, x):
        return torch.where(
            torch.le(x, 0.),
            0.,
            1.,
        )
        return torch.ge(x, 0)


    def forward(self, x):
        val = torch.ones_like(x)
        H1 = self.heaviside(x)
        H2 = self.heaviside(1-x)
        r1 = torch.relu(x)
        r2 = torch.relu(1-x)
        affine = x @ self.weights1 + self.bias
        v = torch.relu(affine) @ self.weights2

        y = 1. + (H1 * r2 - H2 * r1) * v + (r1 * r2) * (self.heaviside(affine) @ (self.weights2 * self.weights1.reshape(self.N, 1)) )

        return x + (r1 * r2) * v, y

    def find_ymin(self, npoints=1024):
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)
        self.K = npoints
        
        _, y = self.forward(x)
        self.ymin = torch.min(y).item()
        return self.ymin


    def project(self, npoints=1024, epsilon=None):
        self.find_ymin(npoints)

        if epsilon is None:
            epsilon = 1e-1
            
        with torch.no_grad():
            if self.ymin < epsilon:
                self.weights2 *= 1 / (1 + epsilon - self.ymin)

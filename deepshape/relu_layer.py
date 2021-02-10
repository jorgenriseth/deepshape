import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from numpy import pi


# Define a Fourier Sine Layer    
class TangentReluLayer1D(nn.Module):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.f1 = nn.Linear(1, N)
        self.f2 = nn.Linear(N, 1)

        self.project()

        #raise NotImplementedError("Proper Derivatives of Tangent Relu Not Implemented.")

        # with torch.no_grad():
        #     self.f1.weight *= 0. + init_scale
        #     self.f2.weight *= 0. + init_scale
        #     self.f1.bias *= 0. + init_scale
        #     self.f2.bias *= 0. + init_scale


    def forward(self, x):
        z = F.relu(self.f1(x))
        y = self.f2(
            torch.where(z >= 0, z, torch.zeros_like(z))
        )
        z = self.f2(z)
        y = 1 + np.pi * torch.cos(np.pi * x) * z + torch.sin(np.pi * x) * y 

        z = x + z * torch.sin(np.pi*x)

        return z, y 

    def find_ymin(self, npoints=1024):
        x = torch.linspace(0, 1, npoints).unsqueeze(-1)
        self.K = npoints
        
        _, y = self.forward(x)
        self.ymin = torch.min(y).item()
        return self.ymin
    
    def project(self, npoints=1024, epsilon=None):
        self.find_ymin(npoints)

        if epsilon is None:
            epsilon = torch.norm(self.f2.weight, 1) * np.pi**3 * self.N / (8 * self.K)
            
        with torch.no_grad():
            if self.ymin < epsilon:
                self.f2.weight *= 1 / (1 + epsilon - self.ymin)




def batch_quadratic(tr, det, epsilon):
    return torch.where(
        torch.logical_or(det < 0, tr <= - 2 * torch.sqrt(det * (1 - epsilon))),
        (- tr - torch.sqrt(tr**2 - (4 * det * (1 - epsilon)))) / (2 * det),
        torch.ones_like(tr)
    )
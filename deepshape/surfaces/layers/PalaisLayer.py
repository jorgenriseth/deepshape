import torch
import torch.nn as nn
from torch import sin, cos

import numpy as np
from numpy import pi

from .DeepShapeLayer import DeepShapeLayer
   
class PalaisLayer(DeepShapeLayer):
    def __init__(self, n, init_scale=0.):
        super().__init__()
        self.eps = torch.finfo(torch.float).eps

        # Number related to basis size
        self.nvec = torch.arange(1, n+1, dtype=torch.float)
        self.n = n
        self.N = 2 * n**2 + n
        
        # Upsampler required in forward pass
        self.upsample = nn.Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = nn.Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )

        self.L = self.lipschitz_vector()

    def forward(self, x):
        """Assumes input on form (K, 2)"""
        # Possible alternative: K = np.prod(x.shape[:-1])
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = sin(pi * z) / self.nvec
        S2 = sin(2 * pi * z)[:, (1, 0), :] / self.nvec
        
        # Cosine matrices
        C1 = cos(pi * z) / self.nvec
        C2 = cos(2 * pi * z)[:, (1, 0), :] / self.nvec
        
        # Tensor product matrices.
        T2 = self.upsample(S1) * S2.repeat(1, 1, n)
        T3 = self.upsample(S1) * C2.repeat(1, 1, n)

        # Function output tensor
        self.B = torch.zeros(K, 2, 2*N)

        self.B[:, 0, :n] = S1[:, 0, :]  # Type 1 x direction
        self.B[:, 1, N:(N+n)] = S1[:, 1, :]  # Type 1  y-direction

        self.B[:, 0, n:(n**2+n)] = T2[:, 0, :]  #Type 2 x-direction
        self.B[:, 1, (N+n):(N + n**2 + n)] = T2[:, 1, :]  # Type 2 y-direction

        self.B[:, 0, (n+n**2):N] = T3[:, 0, :]  # Type 3 x-direction
        self.B[:, 1, (N+n+n**2):] = T3[:, 1, :]  # Type3 y-direction

        return x + self.B @ self.weights

    def lipschitz_vector(self):
        n, N = self.n, self.N 

        upsampled = self.upsample(self.nvec.view(1, 1, -1)).squeeze()
        repeated = self.nvec.repeat(n)
        T23 = (torch.sqrt(upsampled**2 + repeated**2) / (upsampled * repeated)).repeat(2)

        Li = torch.zeros(2*N)
        Li[:n] = 1.
        Li[N:(N+n)] = 1.

        Li[n:N] = T23
        Li[(N+n):] = T23
        return Li * pi

    def project(self):
        with torch.no_grad():
            L = (torch.abs(self.weights) * self.L).sum() #+ self.eps
            if L >= 1:
                self.weights /= L
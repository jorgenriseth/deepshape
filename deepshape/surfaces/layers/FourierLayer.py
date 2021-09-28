from .DeepShapeLayer import DeepShapeLayer
import torch.nn as nn
import torch
from numpy import pi
from torch import cos, sin

class FourierLayer(DeepShapeLayer):
    def __init__(self, n, init_scale=0.):
        super().__init__()

        # Basis Size related numbers
        self.nvec = torch.arange(1, n+1)
        self.n = n
        self.N = 2 * n**2 + n
        
        # Upsampler required in forward pass
        self.upsample = nn.Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = nn.Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )

        
    def forward(self, x):
        """Assumes input on form (K, 2)"""
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = sin(pi * z)
        S2 = sin(2 * pi * z)[:, (1, 0), :]
        
        # Cosine matrices
        C1 = cos(pi * z)
        C2 = cos(2 * pi * z)[:, (1, 0), :]
        
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
    
    def project(self, c=0.9):
        self.weights *= c / self.weights.norm()
import torch
import torch.nn as nn
from torch import sin, cos

import numpy as np
from numpy import pi, sqrt

from .DeepShapeLayer import DeepShapeLayer
from ..utils import torch_square_grid
from ...common import batch_determinant, batch_trace, central_differences, jacobian
   
class PalaisLayer(DeepShapeLayer):
    def __init__(self, n, init_scale=0.0, projection_method="lipschitz"):
        super().__init__()
        self.eps = torch.finfo(torch.float).eps

        # Number related to basis size
        self.n = n
        self.N = 2 * n**2 + n
        
        # Upsampler required in forward pass
        self.upsample = nn.Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = nn.Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )

        # Vectors used function evaluation and projection.
        self.nvec = torch.arange(1, n+1, dtype=torch.float)
        self.L = self.lipschitz_vector()
        
        # Array to be used by some projection methods.
        self.X = torch.tensor([])

        # Properties
        self.device = "cpu"
        self.projection_method = projection_method

        # Ensure initial weights are valid.
        self.project()

    def to(self, device):
        super().to(device)
        self.device = device
        self.nvec = self.nvec.to(device)
        self.L = self.L.to(device)
        self.X = self.X.to(device)
        return self


    def forward(self, x):
        """Assumes input on form (K, 2)"""
        # Possible alternative: K = np.prod(x.shape[:-1])
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = sin(pi * z) / (self.nvec * pi)
        S2 = sin(2 * pi * z)[:, (1, 0), :] / (self.nvec)
        
        # Cosine matrices
        # C1 = cos(pi * z) / (self.nvec * pi)
        C2 = cos(2 * pi * z)[:, (1, 0), :] / (self.nvec) 
        
        # Tensor product matrices.
        T2 = self.upsample(S1) * S2.repeat(1, 1, n)
        T3 = self.upsample(S1) * C2.repeat(1, 1, n)

        # Vector field evaluation.
        self.B = torch.zeros(K, 2, 2*N, device=x.device)

        self.B[:, 0, :n] = S1[:, 0, :]  # Type 1 x direction
        self.B[:, 1, N:(N+n)] = S1[:, 1, :]  # Type 1  y-direction

        self.B[:, 0, n:(n**2+n)] = T2[:, 0, :]  #Type 2 x-direction
        self.B[:, 1, (N+n):(N + n**2 + n)] = T2[:, 1, :]  # Type 2 y-direction

        self.B[:, 0, (n+n**2):N] = T3[:, 0, :]  # Type 3 x-direction
        self.B[:, 1, (N+n+n**2):] = T3[:, 1, :]  # Type3 y-direction

        return x + self.B @ self.weights

    def derivative(self, x, h=None):
        """Assumes input on form (K, 2)"""
        if h is not None:
            return jacobian(self, x, h)

        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = sin(pi * z) / (self.nvec * pi)
        S2 = sin(2 * pi * z)[:, (1, 0), :] / (2 * self.nvec)
        
        # Cosine matrices
        C1 = cos(pi * z) / (self.nvec * pi)
        C2 = cos(2 * pi * z)[:, (1, 0), :] / (2 * self.nvec) 

        # Now for derivative matrices
        T11 = self.upsample(self.nvec * pi * C1) * S2.repeat(1, 1, n)
        T12 = self.upsample(S1) * (2 * pi * self.nvec * C2).repeat(1, 1, n)
        
        T21 = self.upsample(self.nvec * pi * C1) * C2.repeat(1, 1, n)
        T22 = self.upsample(S1) * (-2 * pi * self.nvec * S2).repeat(1, 1, n)
        
        # Create and fill a tensor with derivative outputs
        self.D = torch.zeros(K, 2, 2, 2*N, device=x.device)

        self.D[:, 0, 0, :n] = self.nvec * pi * C1[:, 0, :]  # Type 1 x direction dx 
        self.D[:, 1, 1, N:(N+n)] = self.nvec * pi * C1[:, 1, :]  # Type 1  y-direction dy 
        
        self.D[:, 0, 0, n:(n + n**2)] = T11[:, 0, :] # Type 2 x-direction dx
        self.D[:, 0, 1, n:(n + n**2)] = T12[:, 0, :]  # Type 2 x-direction dy
        self.D[:, 1, 1, (N+n):(N + n + n**2)] = T11[:, 1, :]  # Type 2 y-direction dy
        self.D[:, 1, 0, (N+n):(N + n + n**2)] = T12[:, 1, :]  # Type 2 x-direction dy

        self.D[:, 0, 0, (n+n**2):N] = T21[:, 0, :]  # Type 3 x-direction dx 
        self.D[:, 0, 1, (n+n**2):N] = T22[:, 0, :]  # Type 3 x-direction dy

        self.D[:, 1, 1, (N+n+n**2):] = T21[:, 1, :]  # Type 3 y-direction dy
        self.D[:, 1, 0, (N+n+n**2):] = T22[:, 1, :]  # Type 3 y-direction dx

        # Store trace and determinant for projection.
        self.A = self.D @ self.weights
        
        return torch.eye(2, 2, device=x.device) + self.A

    def lipschitz_vector(self):
        n, N = self.n, self.N 

        upsampled = self.upsample(self.nvec.view(1, 1, -1)).squeeze()
        repeated = self.nvec.repeat(n)
        T23 = (torch.sqrt(4 * upsampled**2 + repeated**2) / (2 * upsampled * repeated)).repeat(2)

        Li = torch.zeros(2*N)
        Li[:n] = 1.
        Li[N:(N+n)] = 1.

        Li[n:N] = T23
        Li[(N+n):] = T23
        return Li

    def project(self, method: str = "lipschitz", **kwargs):
        if self.projection_method == "lipschitz":
            self.project_lipschitz(**kwargs)
        elif self.projection_method == "determinant":
            self.project_determinant(**kwargs)
        elif self.projection_method == "eigen":
            self.project_eigen(**kwargs)
        else:
            raise ValueError(f"Invalid projection method. Got '{self.projection_method}'.")

    def project_lipschitz(self, **kwargs):
        with torch.no_grad():
            L = (torch.abs(self.weights) * self.L).sum() #+ self.eps

            if L >= 1.:
                self.weights /= L


    def project_determinant(self, delta=1e-3, epsilon=1e-2, k=32):
        with torch.no_grad():
            if not hasattr(self, 'D'):
                self.derivative(torch_square_grid(32).reshape(-1, 2))
            trace, det = batch_trace(self.A), batch_determinant(self.A)

            # Compute the smallest root in projection polynomial.
            Q = torch.where(
                torch.abs(trace) < delta,  # Polynomial approximately linear
                batch_linear(trace, epsilon + delta),
                batch_quadratic(trace, det, epsilon) 
            )
            k = Q.min().item()  # Extract smallest root.
            if k < 1.:
                self.weights *= k # Scale weights


    def project_eigen(self, epsilon=1e-3, k=32):
        with torch.no_grad():
            if not hasattr(self, 'D'):
                self.derivative(torch_square_grid(32).reshape(-1, 2))

            trace, det = batch_trace(self.A), batch_determinant(self.A)

            Q = torch.where(
                trace**2 - 2 * det >= 0.,
                torch.ones_like(trace),
                2.0 - epsilon / torch.maximum( (2.0 - epsilon) * torch.ones_like(trace), torch.abs(trace - torch.sqrt(trace**2 - 2 * det)))
            )
            k = float(Q.min())  # Extract smallest root.
            if k < 1.:
                self.weights *= k # Scale weights




# Projection helper function 1 
def batch_quadratic(tr, det, epsilon):
    return torch.where(
        torch.logical_or(det < 0, tr <= - 2 * torch.sqrt(det * (1 - epsilon))),
        (- tr - torch.sqrt(tr**2 - (4 * det * (1 - epsilon)))) / (2 * det),
        torch.ones_like(tr)
    )

# Projection helper function 2
def batch_linear(tr, epsilon):
    return torch.where(
        tr >= - (1 - epsilon),
        torch.tensor([1.], device=tr.device),
        - (1 - epsilon) / tr
    )
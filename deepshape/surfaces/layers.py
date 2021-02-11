import torch
import torch.nn as nn

import numpy as np
from numpy import pi


class FourierLayer(nn.Module):
    """ Implements a Fourier Basis Series layer for 2D diffeomorphisms. 
    Includes basis functions of the form
        sin(n * pi * x)
        sin(n * pi * x) * sin(2 * pi * y)
        sin(n * pi * x) * cos(2 * pi * y)
    in the first component, as well as the same functions with swithced 
    arguments, (x, y) -> (y, x), in the second component.
    """
    def __init__(self, n, init_scale=0.):
        super().__init__()

        # Basis Size related numbers
        self.nvec = torch.arange(1, n+1)
        self.n = n
        self.N = 2 * n**2 + n
        
        # Upsampler required in forward pass
        self.upsample = nn.Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )
        # Ensure positive determinant. 
        #self.project()
    
    
    def forward(self, x):
        """Assumes input on form (K, 2)"""
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = torch.sin(pi * z)
        S2 = torch.sin(2 * pi * z)[:, (1, 0), :]
        
        # Cosine matrices
        C1 = torch.cos(pi * z)
        C2 = torch.cos(2 * pi * z)[:, (1, 0), :]
        
        # Tensor product matrices.
        T2 = self.upsample(S1) * S2.repeat(1, 1, n)
        T3 = self.upsample(S1) * C2.repeat(1, 1, n)

        # Function output tensor
        B = torch.zeros(K, 2, 2*N)

        B[:, 0, :n] = S1[:, 0, :]  # Type 1 x direction
        B[:, 1, N:(N+n)] = S1[:, 1, :]  # Type 1  y-direction

        B[:, 0, n:(n**2+n)] = T2[:, 0, :]  #Type 2 x-direction
        B[:, 1, (N+n):(N + n**2 + n)] = T2[:, 1, :]  # Type 2 y-direction

        B[:, 0, (n+n**2):N] = T3[:, 0, :]  # Type 3 x-direction
        B[:, 1, (N+n+n**2):] = T3[:, 1, :]  # Type3 y-direction
        
        # Now for derivative matrices
        T11 = self.upsample(self.nvec * pi * C1) * S2.repeat(1, 1, n)
        T12 = self.upsample(S1) * (2 * pi * self.nvec * C2).repeat(1, 1, n)
        
        T21 = self.upsample(self.nvec * pi * C1) * C2.repeat(1, 1, n)
        T22 = self.upsample(S1) * (-2 * pi * self.nvec * S2).repeat(1, 1, n)
        
        # Create and fill a tensor with derivative outputs
        D = torch.zeros(K, 2, 2, 2*N)

        D[:, 0, 0, :n] = self.nvec * pi * C1[:, 0, :]  # Type 1 x direction dx 
        D[:, 1, 1, N:(N+n)] = self.nvec * pi * C1[:, 1, :]  # Type 1  y-direction dy 
        
        D[:, 0, 0, n:(n + n**2)] = T11[:, 0, :] # Type 2 x-direction dx
        D[:, 0, 1, n:(n + n**2)] = T12[:, 0, :]  # Type 2 x-direction dy
        D[:, 1, 1, (N+n):(N + n + n**2)] = T11[:, 1, :]  # Type 2 y-direction dy
        D[:, 1, 0, (N+n):(N + n + n**2)] = T12[:, 1, :]  # Type 2 x-direction dy

        D[:, 0, 0, (n+n**2):N] = T21[:, 0, :]  # Type 3 x-direction dx 
        D[:, 0, 1, (n+n**2):N] = T22[:, 0, :]  # Type 3 x-direction dy

        D[:, 1, 1, (N+n+n**2):] = T21[:, 1, :]  # Type 3 y-direction dy
        D[:, 1, 0, (N+n+n**2):] = T22[:, 1, :]  # Type 3 y-direction dx 
        
        # Add a batch-size identity matrix
        I = torch.eye(2).view(1, 2, 2).repeat(K, 1, 1)

        # Store trace and determinant for projection.
        A = D @ self.weights
        self.traceD = batch_trace(A)
        self.detD = batch_determinant(A)

        return x + (B @ self.weights), batch_determinant(I + A)
        
    def project(self, X, epsilon, delta):
       # epsilon = torch.norm(self.weights, 1) * self.n**3 / (8 * 64)
        with torch.no_grad():
            Z, J = self(X)

            # Compute the smallest root in projection polynomial.
            Q = torch.where(
                torch.abs(self.detD) < delta,  # Polynomial approximately linear
                batch_linear(self.traceD, epsilon + delta),
                batch_quadratic(self.traceD, self.detD, epsilon)  # else, set to
            )
            k = Q.min().item()  # Extract smallest root.
            if k < 1.:
                self.weights *= k # Scale weights


class PalaisLayer(nn.Module):
    """ Implements a Fourier Basis Series layer for 2D diffeomorphisms. 
    Includes basis functions of the form
        sin(n * pi * x) / n
        sin(n * pi * x) * sin(2 * m * pi * y) / (nm)
        sin(n * pi * x) * cos(2 * m * pi * y) / (nm)
    in the first component, as well as the same functions with swithced 
    arguments, (x, y) -> (y, x), in the second component.
    """
    def __init__(self, n, init_scale=0.):
        super().__init__()

        # Basis Size related numbers
        self.nvec = torch.arange(1, n+1)
        self.n = n
        self.N = 2 * n**2 + n
        
        # Upsampler required in forward pass
        self.upsample = nn.Upsample(scale_factor=n, mode='nearest')

        # Create weight vector
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(2*self.N, requires_grad=True)
        )
        # Ensure positive determinant. 
        #self.project()
    
    
    def forward(self, x):
        """Assumes input on form (K, 2)"""
        K = x.shape[0]
        n, N = self.n, self.N
        z = (x.view(K, 2, 1) * self.nvec)
        
        # Sine matrices
        S1 = torch.sin(pi * z) / self.nvec
        S2 = torch.sin(2 * pi * z)[:, (1, 0), :] / self.nvec
        
        # Cosine matrices
        C1 = torch.cos(pi * z) / self.nvec
        C2 = torch.cos(2 * pi * z)[:, (1, 0), :] / self.nvec
        
        # Tensor product matrices.
        T2 = self.upsample(S1) * S2.repeat(1, 1, n)
        T3 = self.upsample(S1) * C2.repeat(1, 1, n)

        # Function output tensor
        B = torch.zeros(K, 2, 2*N)

        B[:, 0, :n] = S1[:, 0, :]  # Type 1 x direction
        B[:, 1, N:(N+n)] = S1[:, 1, :]  # Type 1  y-direction

        B[:, 0, n:(n**2+n)] = T2[:, 0, :]  #Type 2 x-direction
        B[:, 1, (N+n):(N + n**2 + n)] = T2[:, 1, :]  # Type 2 y-direction

        B[:, 0, (n+n**2):N] = T3[:, 0, :]  # Type 3 x-direction
        B[:, 1, (N+n+n**2):] = T3[:, 1, :]  # Type3 y-direction
        
        # Now for derivative matrices
        T11 = self.upsample(self.nvec * pi * C1) * S2.repeat(1, 1, n)
        T12 = self.upsample(S1) * (2 * pi * self.nvec * C2).repeat(1, 1, n)
        
        T21 = self.upsample(self.nvec * pi * C1) * C2.repeat(1, 1, n)
        T22 = self.upsample(S1) * (-2 * pi * self.nvec * S2).repeat(1, 1, n)
        
        # Create and fill a tensor with derivative outputs
        D = torch.zeros(K, 2, 2, 2*N)

        D[:, 0, 0, :n] = self.nvec * pi * C1[:, 0, :]  # Type 1 x direction dx 
        D[:, 1, 1, N:(N+n)] = self.nvec * pi * C1[:, 1, :]  # Type 1  y-direction dy 
        
        D[:, 0, 0, n:(n + n**2)] = T11[:, 0, :] # Type 2 x-direction dx
        D[:, 0, 1, n:(n + n**2)] = T12[:, 0, :]  # Type 2 x-direction dy
        D[:, 1, 1, (N+n):(N + n + n**2)] = T11[:, 1, :]  # Type 2 y-direction dy
        D[:, 1, 0, (N+n):(N + n + n**2)] = T12[:, 1, :]  # Type 2 x-direction dy

        D[:, 0, 0, (n+n**2):N] = T21[:, 0, :]  # Type 3 x-direction dx 
        D[:, 0, 1, (n+n**2):N] = T22[:, 0, :]  # Type 3 x-direction dy

        D[:, 1, 1, (N+n+n**2):] = T21[:, 1, :]  # Type 3 y-direction dy
        D[:, 1, 0, (N+n+n**2):] = T22[:, 1, :]  # Type 3 y-direction dx 
        
        # Add a batch-size identity matrix
        I = torch.eye(2).view(1, 2, 2).repeat(K, 1, 1)

        # Store trace and determinant for projection.
        A = D @ self.weights
        self.traceD = batch_trace(A)
        self.detD = batch_determinant(A)

        return x + (B @ self.weights), batch_determinant(I + A)
        
    def project(self, X, epsilon, delta):
       # epsilon = torch.norm(self.weights, 1) * self.n**3 / (8 * 64)
        with torch.no_grad():
            Z, J = self(X)

            # Compute the smallest root in projection polynomial.
            Q = torch.where(
                torch.abs(self.detD) < delta,  # Polynomial approximately linear
                batch_linear(self.traceD, epsilon + delta),
                batch_quadratic(self.traceD, self.detD, epsilon)  # else, set to
            )
            k = Q.min().item()  # Extract smallest root.
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
        torch.Tensor([1.]),
        - (1 - epsilon) / tr
    )


# Compute determinant of a batch of 2x2 matrices.
def batch_determinant(B):
    assert B.dim() == 3, f"Dim.shape should be (K, 2, 2), got {B.shape}"
    return (B[:, 0, 0] * B[:, 1, 1] - B[:, 1, 0] * B[:, 0, 1]).view(-1, 1)


# Compute trace of a batch of 2x2 matrices.
def batch_trace(B):
    assert B.dim() == 3, f"Dim.shape should be (K, 2, 2), got {B.shape}"
    return (B[:, 0, 0] + B[:, 1, 1]).view(-1, 1)

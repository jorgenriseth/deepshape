import torch
import torch.nn as nn
import numpy as np

from torch import cos, sin
from numpy import pi

from ..common import central_differences


class CurveApprox(nn.Module):
    def derivative(self, X, h=5e-4):
        """ Finite difference approximation of curve velocity. """
        return (0.5 / h) * (self.forward(X + h) - self.forward(X - h) ) / h
    

class  FourierApprox(CurveApprox):
    def __init__(self, Xtr: torch.Tensor, y: torch.Tensor, N: int):
        super().__init__()
        self.N = N
        self.dim = 1 + 2*N
        self.freqvec = torch.arange(1, N+1) * 2.0 * pi
        self.find_weights(Xtr, y)

    def eval_basis(self, x):
        self.check_input(x)
        T = self.freqvec * x
        B = torch.cat([torch.ones_like(x), sin(T), cos(T)], dim=1)
        return B

    def eval_basis_derivative(self, x):
        self.check_input(x)
        T = self.freqvec * x
        dB = torch.cat([torch.zeros_like(x), self.freqvec * cos(T), -self.freqvec * sin(T)], dim=1)
        return dB

    def forward(self, x):
        B = self.eval_basis(x)
        return torch.mm(B, self.w)
        
    def derivative(self, x):
        dB = self.eval_basis_derivative(x)
        return torch.mm(dB, self.w)

    def find_weights(self, x, y):
        B = self.eval_basis(x)
        w = np.linalg.solve(torch.mm(B.t(), B).numpy(),
                            torch.mm(B.t(), y).numpy())
        self.w = torch.Tensor(w)

    def check_input(self, y):
        if not (y.dim() == 2 and y.size()[1] == 1):
            raise ValueError("y should be 2D-Tensor")


class QmapApprox(CurveApprox):
    def __init__(self, fa : FourierApprox):
        # centroid = fa.compute_centroid()
        # curvelen = fa.compute_curvelen()
        super().__init__()
        self.fa = fa

    def forward(self, x, h=None):
        if h is None:
            return torch.sqrt(self.fa.derivative(x).norm(dim=-1, keepdim=True)) * self.fa(x)
        return torch.sqrt(central_differences(self.fa, x, h).norm(dim=-1, keepdim=True)) * self.fa(x)




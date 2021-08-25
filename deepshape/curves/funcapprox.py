import torch
import torch.nn as nn
import numpy as np

from torch import cos, sin
from numpy import pi

class  FourierApprox(nn.Module):
    def __init__(self, Xtr: torch.Tensor, y: torch.Tensor, N: int):
        super().__init__()
        self.N = N
        self.dim = 1 + 2*N
        self.Nvec = torch.arange(1, N+1)
        self.approx(Xtr, y)

    def eval_basis(self, x):
        self.check_input(x)
        T = self.Nvec * x * 2.0 * pi
        B = torch.cat([torch.ones_like(x), sin(T), cos(T)], dim=1)
        return B

    def forward(self, x):
        B = self.eval_basis(x)
        return torch.mm(B, self.w)

    def approx(self, x, y):
        B = self.eval_basis(x)
        w = np.linalg.solve(torch.mm(B.t(), B).numpy(),
                            torch.mm(B.t(), y).numpy())
        self.w = torch.Tensor(w)

    def check_input(self, y):
        if not (y.dim() == 2 and y.size()[1] == 1):
            raise ValueError("y should be 2D-Tensor")


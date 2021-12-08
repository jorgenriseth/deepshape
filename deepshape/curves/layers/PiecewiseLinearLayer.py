import torch
import numpy as np
from torch.nn import Parameter

from .DeepShapeLayer import DeepShapeLayer


class PiecewiseLinearLayer(DeepShapeLayer):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.h = 1 / (N + 1)

        self.w = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.nodes = torch.linspace(0, 1, N+2)
        self.project()

    def forward(self, x):
        Z = (torch.relu(x - self.nodes[:-2]) 
            - 2 * torch.relu(x - self.nodes[1:-1])
            + torch.relu(x - self.nodes[2:]))
        return x + Z @ self.w / self.h

    def project(self):
        with torch.no_grad():
            if self.N != 1:
                k = torch.max(self.w[:-1] - self.w[1:])
            else:
                k = self.h

            k = max(k, -self.w[0], self.w[-1] - 1.)

            if k > self.h:
                self.w *= (self.h / k * 0.99)
            # norm = self.w.norm(p=1) / self.h
            # if norm > 0.99:
            #     self.w /= norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self

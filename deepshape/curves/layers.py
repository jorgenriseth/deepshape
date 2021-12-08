import torch
from numpy import pi, sqrt

from ..common import CurveLayer


class SineSeries(CurveLayer):
    def __init__(self, N, init_scale=0.):
        super().__init__()
        self.N = N
        self.nvec = torch.arange(1, N+1, dtype=torch.float)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(N, 1, requires_grad=True)
        )
        self.project(p=1)

    def forward(self, x):
        return x + (torch.sin(pi * self.nvec * x) / (pi * self.nvec)) @ self.weights

    def derivative(self, x, h=None):
        return 1. + torch.cos(pi * self.nvec * x) @ self.weights

    def project(self, p=1):
        with torch.no_grad():
            if p == 1:
                norm = self.weights.norm(p)
            elif p == float('inf'):
                norm = torch.abs(self.weights).max() * self.N
            elif p == 2:
                norm = self.weights.norm(p) * sqrt(self.N)
            else:
                q = p / (p - 1)
                norm = self.weights.norm(p) * torch.ones(self.N).norm(q)
            if norm > 1.0 - 1e-4:
                self.weights *= (1 - 1e-4) / norm

    def to(self, device):
        self.nvec = self.nvec.to(device)
        return self


# TODO: Consider removal, as this is a worse version of its parent
class UnscaledSineSeries(SineSeries):
    def forward(self, x):
        return x + (torch.sin(pi * self.nvec * x) / (pi * self.nvec)) @ self.weights

    def derivative(self, x, h=None):
        return 1. + pi * (self.nvec * torch.cos(pi * self.nvec * x)) @ self.weights

    def project(self, p=1):
        with torch.no_grad():
            if p == 1:
                norm = (self.weights * pi * self.nvec).norm(p)
            elif p == float('inf'):
                norm = torch.abs(self.weights).max() * pi * \
                    (0.5 * self.N * (self.N + 1))
            elif p == 2:
                norm = (self.weights.norm(p)
                        * pi * sqrt(self.N * (self.N + 1) * (2 * self.N + 1) / 6))
            else:
                q = p / (p - 1)
                norm = self.weights.norm(p) * torch.ones(self.nvec).norm(q)
            if norm > 1.0 - 1e-4:
                self.weights *= (1 - 1e-4) / norm


class PalaisLayer(SineSeries):
    def __init__(self, N, init_scale=0.):
        super().__init__(N)
        self.weights = torch.nn.Parameter(
            init_scale * torch.randn(2*N, 1, requires_grad=True)
        )
        self.project(p=1)

    def forward(self, x):
        y = torch.cat([
            torch.sin(pi * self.nvec * x) / (pi * self.nvec),
            (torch.cos(2*pi * self.nvec * x) - 1.) / (2*pi * self.nvec)
        ], axis=-1)
        return x + y @ self.weights

    def derivative(self, x, h=None):
        u = torch.cat([
            torch.cos(2*pi * self.nvec * x),
            -torch.sin(2*pi * self.nvec * x)
        ], axis=-1)
        return 1 + u @ self.weights

    def project(self, p=1):
        assert p == 1, f"PalaisLayer projection only defined for p=1, got {p}"
        super().project(p)


class HatLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()


class SpectralDenseLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()


class BumpReluLayer(CurveLayer):
    def __init__(self):
        return NotImplementedError()
